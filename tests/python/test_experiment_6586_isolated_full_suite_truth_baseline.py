"""Focused tests for the isolated repository-wide Python suite baseline.

Spec refs: REQ-REPORT-6586, SCENARIO-REPORT-6586-DISPOSABLE,
SCENARIO-REPORT-6586-DIRTY-OVERLAY, SCENARIO-REPORT-6586-RED,
SCENARIO-REPORT-6586-TIMEOUT, SCENARIO-REPORT-6586-MUTATION,
SCENARIO-REPORT-6586-ATTACKS, SCENARIO-REPORT-6586-ATOMIC.
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

from carnot import experiment_6586_isolated_full_suite_truth_baseline as exp


@pytest.fixture
def repo_root() -> Path:
    """Return the active repository for spec-only assertions."""

    return Path(__file__).resolve().parents[2]


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=root, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def _init_repo(root: Path) -> None:
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "user.name", "Test User")


def _base_evidence(tmp_path: Path) -> dict[str, object]:
    checkout = tmp_path / "carnot-exp6586-test" / "checkout"
    checkout.mkdir(parents=True)
    collection = {
        "command": exp.COLLECTION_COMMAND_TEXT,
        "cwd": str(checkout),
        "exit_code": 0,
        "timed_out": False,
        "duration_s": 1.0,
        "stdout": "10 tests collected\n",
        "stderr": "",
        "collected_count": 10,
        "nodeids_sha256": exp.sha256_json([f"test_{i}" for i in range(10)]),
        "receipt_sha256": "sha256:collection",
        "process_cleanup": {
            "clean": True,
            "surviving_owned_pids": [],
            "unrelated_process_signal_count": 0,
        },
    }
    suite = {
        "command": exp.SUITE_COMMAND_TEXT,
        "argv": list(exp.SUITE_COMMAND),
        "cwd": str(checkout),
        "environment_sha256": "sha256:environment",
        "exit_code": 1,
        "timed_out": False,
        "duration_s": 2.0,
        "stdout": "1 failed, 9 passed\n",
        "stderr": "",
        "collected_count": 10,
        "process_cleanup": {
            "clean": True,
            "owned_process_group": 12345,
            "signals": [],
            "surviving_owned_pids": [],
            "unrelated_process_signal_count": 0,
        },
    }
    rows = [
        {
            "nodeid": "tests/python/test_sample.py::test_bad",
            "outcome": "failed",
            "phase": "call",
            "longrepr": "assert 1 == 2",
        }
    ]
    checkout_receipt = {
        "active_root": str(tmp_path / "active"),
        "checkout_root": str(checkout),
        "validated_temporary_root": str(checkout.parent),
        "revision": "a" * 40,
        "detached_head": True,
        "patch_hash": exp.sha256_json([]),
        "patch_rows": [],
        "dirty_paths": [],
        "overlay_complete": True,
        "before_tracked_snapshot_sha256": "sha256:before",
        "after_tracked_snapshot_sha256": "sha256:after",
        "mutation_scan_complete": True,
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
        "collection": collection,
        "suite": suite,
        "rows": rows,
        "checkout": checkout_receipt,
        "active": active,
        "protected": protected,
    }


def test_req_report_6586_spec_and_command_are_exact(repo_root: Path) -> None:
    """REQ-REPORT-6586 binds the serial no-coverage command in the spec."""

    spec = (repo_root / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )
    assert "REQ-REPORT-6586" in spec
    assert "SCENARIO-REPORT-6586-DISPOSABLE" in spec
    assert exp.SUITE_COMMAND_TEXT == (
        ".venv/bin/python -m pytest tests/python --no-cov -o addopts= -n 0"
    )
    assert exp.INFERENCE_SUBSTRATE == "isolated_repo_test_execution_no_llm"
    argv = exp._actual_argv(repo_root, exp.SUITE_COMMAND)
    assert argv[0] == str(repo_root / ".venv/bin/python")


def test_req_report_6586_temporary_root_refuses_broad_and_unsafe_paths(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-6586-DISPOSABLE accepts only a narrow temp path."""

    active = tmp_path / "active"
    active.mkdir()
    narrow = tmp_path / "carnot-exp6586-owned"
    narrow.mkdir()
    assert exp.validate_temporary_root(narrow, active, temp_root=tmp_path) == narrow.resolve()

    for unsafe in (tmp_path, active, active / "child", tmp_path.parent):
        if unsafe == active / "child":
            unsafe.mkdir()
        with pytest.raises(exp.IsolationError):
            exp.validate_temporary_root(unsafe, active, temp_root=tmp_path)

    missing = tmp_path / "missing"
    with pytest.raises(exp.IsolationError, match="must exist"):
        exp.validate_temporary_root(missing, active, temp_root=tmp_path)
    link = tmp_path / "carnot-exp6586-link"
    link.symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(exp.IsolationError):
        exp.validate_temporary_root(link, active, temp_root=tmp_path)


def test_req_report_6586_dirty_overlay_copies_every_active_change(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6586-DIRTY-OVERLAY hashes writes and deletions."""

    active = tmp_path / "active"
    active.mkdir()
    _init_repo(active)
    (active / "keep.txt").write_text("old\n", encoding="utf-8")
    (active / "delete.txt").write_text("delete\n", encoding="utf-8")
    _git(active, "add", ".")
    _git(active, "commit", "-qm", "base")

    checkout = tmp_path / "checkout"
    subprocess.run(
        ["git", "clone", "-q", str(active), str(checkout)], check=True, capture_output=True
    )
    (active / "keep.txt").write_text("new\n", encoding="utf-8")
    (active / "delete.txt").unlink()
    (active / "new.txt").write_text("untracked\n", encoding="utf-8")

    dirty = exp.active_dirty_paths(active)
    assert dirty == ["delete.txt", "keep.txt", "new.txt"]
    rows = exp.apply_content_overlay(active, checkout, dirty)
    assert exp.overlay_is_complete(dirty, rows, active, checkout)
    assert (checkout / "keep.txt").read_text(encoding="utf-8") == "new\n"
    assert not (checkout / "delete.txt").exists()
    assert (checkout / "new.txt").read_text(encoding="utf-8") == "untracked\n"
    assert {row["action"] for row in rows} == {"write", "delete"}
    assert all(row["row_sha256"] == exp.overlay_row_hash(row) for row in rows)
    assert not exp.overlay_is_complete(dirty, rows[:-1], active, checkout)


def test_req_report_6586_hash_snapshots_name_all_mutations(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6586-MUTATION keeps each changed tracked path."""

    root = tmp_path / "repo"
    root.mkdir()
    _init_repo(root)
    (root / "same.txt").write_text("same", encoding="utf-8")
    (root / "changed.txt").write_text("before", encoding="utf-8")
    (root / "deleted.txt").write_text("delete", encoding="utf-8")
    _git(root, "add", ".")
    _git(root, "commit", "-qm", "base")
    before = exp.snapshot_tracked_files(root)
    (root / "changed.txt").write_text("after", encoding="utf-8")
    (root / "deleted.txt").unlink()
    after = exp.snapshot_tracked_files(root)
    rows = exp.tracked_mutation_rows(before, after)
    assert [row["path"] for row in rows] == ["changed.txt", "deleted.txt"]
    assert rows[0]["before_hash"] != rows[0]["after_hash"]
    assert rows[1]["after_hash"] is None
    assert exp.snapshot_checksum(before).startswith("sha256:")
    curated = exp.operator_curated_snapshot(root, patterns=("*.txt",))
    assert set(curated) == set(after)
    assert curated["changed.txt"]["content_hash"] == exp.hash_path(root / "changed.txt")


def test_req_report_6586_pytest_plugin_preserves_rows_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-6586-ROWS records failure, error, skip, and collection data."""

    receipt = tmp_path / "plugin.json"
    monkeypatch.setenv(exp.PLUGIN_RECEIPT_ENV, str(receipt))
    exp._reset_plugin_state()
    items = [SimpleNamespace(nodeid="a::test_one"), SimpleNamespace(nodeid="b::test_two")]
    exp.pytest_collection_finish(SimpleNamespace(items=items))
    exp.pytest_collectreport(
        SimpleNamespace(failed=True, nodeid="tests/python/test_bad.py", longrepr="import error")
    )
    for outcome, phase in (("failed", "call"), ("skipped", "setup")):
        exp.pytest_runtest_logreport(
            SimpleNamespace(
                passed=False,
                failed=outcome == "failed",
                skipped=outcome == "skipped",
                nodeid=f"x::{outcome}",
                when=phase,
                longrepr=f"{outcome} detail",
                wasxfail=None,
            )
        )
    exp.pytest_runtest_logreport(
        SimpleNamespace(
            passed=True,
            failed=False,
            skipped=False,
            nodeid="x::pass",
            when="call",
            longrepr="",
            wasxfail=None,
        )
    )
    exp.pytest_sessionfinish(SimpleNamespace(), 1)
    payload = json.loads(receipt.read_text(encoding="utf-8"))
    assert payload["collected_count"] == 2
    assert {row["outcome"] for row in payload["rows"]} == {
        "errored",
        "failed",
        "skipped",
    }
    assert not list(tmp_path.glob(".plugin.json.*.tmp"))


def test_req_report_6586_owned_command_records_success_and_timeout(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6586-TIMEOUT cleans only the owned process group."""

    success = exp.run_owned_command(
        [sys.executable, "-c", "print('ok')"],
        cwd=tmp_path,
        env=os.environ.copy(),
        timeout_s=5.0,
        display_command="python success",
        cleanup_grace_s=0.05,
    )
    assert success["exit_code"] == 0
    assert success["stdout"].strip() == "ok"
    assert success["process_cleanup"]["clean"] is True
    assert success["process_cleanup"]["unrelated_process_signal_count"] == 0

    timeout = exp.run_owned_command(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        cwd=tmp_path,
        env=os.environ.copy(),
        timeout_s=0.05,
        display_command="python timeout",
        cleanup_grace_s=0.05,
    )
    assert timeout["timed_out"] is True
    assert timeout["exit_code"] is not None
    assert timeout["process_cleanup"]["clean"] is True
    assert timeout["process_cleanup"]["signals"][0]["signal"] == "SIGTERM"


def test_req_report_6586_red_baseline_is_complete_null_infrastructure(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-6586-RED treats failures as a complete measurement."""

    evidence = _base_evidence(tmp_path)
    decision = exp.reduce_suite_truth(
        collection=evidence["collection"],
        suite=evidence["suite"],
        rows=evidence["rows"],
        mutation_rows=[],
        checkout=evidence["checkout"],
        active_unchanged=evidence["active"],
    )
    assert decision["state"] == "measured_red"
    assert decision["complete"] is True
    assert decision["ready_score"] == 1
    assert decision["verdict_class"] == "null"

    report = exp.build_report(
        run_date="20260825",
        preconditions={"inference_substrate": exp.INFERENCE_SUBSTRATE},
        checkout=evidence["checkout"],
        collection=evidence["collection"],
        suite=evidence["suite"],
        rows=evidence["rows"],
        mutation_rows=[],
        active_unchanged=evidence["active"],
        protected=evidence["protected"],
        tests_run=[{"command": "focused", "exit_code": 0}],
        duration_s=3.0,
    )
    assert report["status"] == "measured_red"
    assert report["honest_verdict"].startswith("complete:")
    assert report["verdict_class"] == "null"
    assert report["full_suite_baseline_ready_score"] == 1
    assert report["low_cadence_ownership_contract"]["experiment_launch_gate"] is False
    assert report["low_cadence_ownership_contract"]["owner"] == "repository_maintainer"
    assert report["family_summaries"][0]["failed"] == 1
    assert exp.validate_report(report) == []


def test_req_report_6586_green_timeout_and_environment_block_reduce_honestly(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-6586-VERDICT separates complete, timeout, and blocked states."""

    evidence = _base_evidence(tmp_path)
    green_suite = deepcopy(evidence["suite"])
    green_suite["exit_code"] = 0
    green_suite["stdout"] = "10 passed"
    green = exp.reduce_suite_truth(
        collection=evidence["collection"],
        suite=green_suite,
        rows=[],
        mutation_rows=[],
        checkout=evidence["checkout"],
        active_unchanged=evidence["active"],
    )
    assert green["state"] == "measured_green"

    timeout_suite = deepcopy(green_suite)
    timeout_suite["timed_out"] = True
    timeout_suite["exit_code"] = -15
    timeout = exp.reduce_suite_truth(
        collection=evidence["collection"],
        suite=timeout_suite,
        rows=[exp.timeout_row(30.0)],
        mutation_rows=[],
        checkout=evidence["checkout"],
        active_unchanged=evidence["active"],
    )
    assert timeout["state"] == "timeout"
    assert timeout["ready_score"] == 0
    assert timeout["verdict_class"] == "partial"

    blocked = exp.blocked_report(
        run_date="20260825",
        failed_check="temporary_root",
        observed_value="/tmp",
        duration_s=0.1,
    )
    assert blocked["status"] == "isolated_environment_block"
    assert blocked["honest_verdict"].startswith("blocked_")
    assert blocked["verdict_class"] == "blocked"
    assert blocked["gate_check_summary"][0]["observed_value"] == "/tmp"
    assert exp.validate_report(blocked) == []


def test_req_report_6586_attacks_and_validation_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6586-ATTACKS rejects all seven false baselines."""

    evidence = _base_evidence(tmp_path)
    report = exp.build_report(
        run_date="20260825",
        preconditions={"inference_substrate": exp.INFERENCE_SUBSTRATE},
        checkout=evidence["checkout"],
        collection=evidence["collection"],
        suite=evidence["suite"],
        rows=evidence["rows"],
        mutation_rows=[],
        active_unchanged=evidence["active"],
        protected=evidence["protected"],
        tests_run=[],
        duration_s=3.0,
    )
    assert [row["attack"] for row in report["attack_rows"]] == list(exp.REQUIRED_ATTACKS)
    assert all(row["passed"] for row in report["attack_rows"])

    mutations = {
        "passing_headline_with_failed_rows": ("status", "measured_green"),
        "timeout_called_green": ("suite_command_receipt", {**evidence["suite"], "timed_out": True}),
        "unreported_tracked_write": ("mutation_rows", [{"path": "changed"}]),
        "leaked_child_process": (
            "suite_command_receipt",
            {
                **evidence["suite"],
                "process_cleanup": {
                    **evidence["suite"]["process_cleanup"],
                    "clean": False,
                    "surviving_owned_pids": [999],
                },
            },
        ),
        "active_tree_hash_drift": (
            "active_worktree_unchanged",
            {**evidence["active"], "unchanged": False},
        ),
    }
    for name, (field, value) in mutations.items():
        broken = deepcopy(report)
        broken[field] = value
        broken["reproducibility_checksum"] = exp.artifact_checksum(broken)
        assert exp.validate_report(broken), name

    missing = deepcopy(report)
    missing.pop("rows")
    assert "missing_required_field:rows" in exp.validate_report(missing)


def test_req_report_6586_atomic_checksum_and_defensive_helpers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-REPORT-6586-ATOMIC syncs one checksummed terminal artifact."""

    evidence = _base_evidence(tmp_path)
    report = exp.build_report(
        run_date="20260825",
        preconditions={"inference_substrate": exp.INFERENCE_SUBSTRATE},
        checkout=evidence["checkout"],
        collection=evidence["collection"],
        suite=evidence["suite"],
        rows=evidence["rows"],
        mutation_rows=[],
        active_unchanged=evidence["active"],
        protected=evidence["protected"],
        tests_run=[],
        duration_s=3.0,
    )
    target = tmp_path / "result.json"
    receipt = exp.atomic_write_report(target, report)
    assert receipt["atomic_replace"] is True
    assert receipt["file_fsync"] is True
    assert receipt["directory_fsync"] is True
    assert json.loads(target.read_text(encoding="utf-8")) == report
    assert exp.hash_path(tmp_path / "absent") is None

    bad = deepcopy(report)
    bad["reproducibility_checksum"] = "sha256:bad"
    with pytest.raises(ValueError, match="reproducibility_checksum"):
        exp.atomic_write_report(tmp_path / "bad.json", bad)

    monkeypatch.delenv(exp.PLUGIN_RECEIPT_ENV, raising=False)
    exp._reset_plugin_state()
    exp.pytest_sessionfinish(SimpleNamespace(), 0)


def test_req_report_6586_helper_error_and_symlink_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-6586-CHECKOUT rejects unsafe paths and preserves links."""

    active = tmp_path / "active"
    checkout = tmp_path / "checkout"
    active.mkdir()
    checkout.mkdir()
    target = active / "target.txt"
    target.write_text("target", encoding="utf-8")
    (active / "link.txt").symlink_to("target.txt")
    rows = exp.apply_content_overlay(active, checkout, ["link.txt"])
    assert rows[0]["kind"] == "symlink"
    assert (checkout / "link.txt").is_symlink()
    assert exp.hash_path(active / "link.txt") == exp.hash_path(checkout / "link.txt")
    for unsafe in ("../escape", "/absolute"):
        with pytest.raises(exp.IsolationError, match="escape"):
            exp.apply_content_overlay(active, checkout, [unsafe])

    file_root = tmp_path / "file-root"
    file_root.write_text("not a directory", encoding="utf-8")
    with pytest.raises(exp.IsolationError, match="directory"):
        exp.validate_temporary_root(file_root, active, temp_root=tmp_path)
    with pytest.raises(exp.IsolationError, match="git could not answer"):
        exp.git_revision(active)
    with pytest.raises(exp.IsolationError, match="valid receipt"):
        exp._read_plugin_receipt(tmp_path / "missing.json")
    invalid = tmp_path / "invalid.json"
    invalid.write_text("[]", encoding="utf-8")
    with pytest.raises(exp.IsolationError, match="schema"):
        exp._read_plugin_receipt(invalid)

    bad_rows = deepcopy(rows)
    bad_rows[0]["row_sha256"] = "sha256:bad"
    assert not exp.overlay_is_complete(["link.txt"], bad_rows, active, checkout)
    (checkout / "link.txt").unlink()
    (checkout / "link.txt").write_text("different", encoding="utf-8")
    assert not exp.overlay_is_complete(["link.txt"], rows, active, checkout)

    monkeypatch.setenv("PYTHONPATH", "existing")
    env, receipt = exp._effective_environment(
        checkout, tmp_path / "owned", tmp_path / "receipt.json", "run-id"
    )
    assert env["PYTHONPATH"].endswith("existing")
    assert receipt["sha256"] == exp.sha256_json(receipt["values"])


def test_req_report_6586_plugin_priority_observation_and_process_edges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-6586-ROWS and TIMEOUT retain strongest outcomes and cleanup."""

    exp.pytest_sessionstart(SimpleNamespace())
    exp.pytest_runtest_logreport(
        SimpleNamespace(
            passed=False,
            failed=True,
            skipped=False,
            nodeid="tests/python/samplers/test_x.py::test_error",
            when="teardown",
            longrepr="teardown error",
            wasxfail=None,
        )
    )
    exp.pytest_runtest_logreport(
        SimpleNamespace(
            passed=False,
            failed=False,
            skipped=False,
            nodeid="ignored",
            when="call",
            longrepr="",
            wasxfail=None,
        )
    )
    rows, summaries = exp._plugin_terminal_rows()
    assert rows[0]["outcome"] == "errored"
    assert summaries[0]["family"] == "samplers"

    checkout = tmp_path / "checkout"
    log_root = checkout / "ops/.test_suite_mutation_runs"
    log_root.mkdir(parents=True)
    inside = checkout / "tracked.txt"
    outside = tmp_path / "outside.txt"
    (log_root / "one.writes.log").write_text(f"{inside}\n{outside}\n", encoding="utf-8")
    assert exp._observed_write_paths(checkout, ["missing", "one"]) == ["tracked.txt"]

    exp._signal_owned_group(999_999_999, __import__("signal").SIGTERM, [])
    ignored_term = exp.run_owned_command(
        [
            sys.executable,
            "-c",
            "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)",
        ],
        cwd=tmp_path,
        env=os.environ.copy(),
        timeout_s=0.05,
        display_command="ignore term",
        cleanup_grace_s=0.02,
    )
    assert [row["signal"] for row in ignored_term["process_cleanup"]["signals"]] == [
        "SIGTERM",
        "SIGKILL",
    ]

    code = (
        "import subprocess,sys; "
        "subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],"
        "stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)"
    )
    leaked = exp.run_owned_command(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=os.environ.copy(),
        timeout_s=2.0,
        display_command="leaked child",
        cleanup_grace_s=0.05,
    )
    assert leaked["timed_out"] is False
    assert leaked["process_cleanup"]["leaked_owned_pids_before_cleanup"]
    assert leaked["process_cleanup"]["clean"] is False

    monkeypatch.delenv("PYTHONPATH", raising=False)
    env, _receipt = exp._effective_environment(
        checkout, tmp_path / "owned-two", tmp_path / "receipt-two.json", "run-two"
    )
    assert "existing" not in env["PYTHONPATH"]


def test_req_report_6586_validation_covers_every_refusal(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6586-ATTACKS exercises each validation refusal branch."""

    evidence = _base_evidence(tmp_path)
    report = exp.build_report(
        run_date="20260825",
        preconditions={"inference_substrate": exp.INFERENCE_SUBSTRATE},
        checkout=evidence["checkout"],
        collection=evidence["collection"],
        suite=evidence["suite"],
        rows=evidence["rows"],
        mutation_rows=[],
        active_unchanged=evidence["active"],
        protected=evidence["protected"],
        tests_run=[],
        duration_s=3.0,
    )
    cases = []
    for field, value in (
        ("inference_substrate", "wrong"),
        ("verifier_is_oracle", False),
        ("honest_verdict", "red without prefix"),
        ("attack_rows", []),
        ("low_cadence_ownership_contract", {"experiment_launch_gate": True}),
        ("protected_files_unchanged", {"unchanged": False, "before": {}, "after": {}}),
    ):
        broken = deepcopy(report)
        broken[field] = value
        broken["reproducibility_checksum"] = exp.artifact_checksum(broken)
        cases.append(broken)
    for broken in cases:
        assert exp.validate_report(broken)

    for suite_change in (
        {"command": "wrong"},
        {"cwd": "/active"},
        {
            "process_cleanup": {
                "clean": True,
                "surviving_owned_pids": [],
                "unrelated_process_signal_count": 1,
            }
        },
        {"collected_count": 9},
    ):
        broken = deepcopy(report)
        broken["suite_command_receipt"].update(suite_change)
        broken["reproducibility_checksum"] = exp.artifact_checksum(broken)
        assert exp.validate_report(broken)
    broken = deepcopy(report)
    broken["collection_receipt"]["command"] = "wrong"
    broken["reproducibility_checksum"] = exp.artifact_checksum(broken)
    assert exp.validate_report(broken)
    broken = deepcopy(report)
    broken["collection_receipt"]["cwd"] = "/active"
    broken["reproducibility_checksum"] = exp.artifact_checksum(broken)
    assert exp.validate_report(broken)
    broken = deepcopy(report)
    broken["collection_receipt"]["process_cleanup"] = {
        "clean": False,
        "surviving_owned_pids": [8],
        "unrelated_process_signal_count": 1,
    }
    broken["reproducibility_checksum"] = exp.artifact_checksum(broken)
    errors = exp.validate_report(broken)
    assert "collection_process_leak" in errors
    assert "collection_unrelated_process_signaled" in errors
    broken = deepcopy(report)
    broken["rows"] = [{"nodeid": "x", "outcome": "passed"}]
    broken["reproducibility_checksum"] = exp.artifact_checksum(broken)
    assert exp.validate_report(broken)

    blocked = exp.blocked_report(
        run_date="20260825", failed_check="x", observed_value=False, duration_s=0.1
    )
    for field, value in (
        ("verdict_class", "null"),
        ("honest_verdict", "complete: wrong"),
        ("gate_check_summary", []),
    ):
        broken = deepcopy(blocked)
        broken[field] = value
        broken["reproducibility_checksum"] = exp.artifact_checksum(broken)
        assert exp.validate_report(broken)


@pytest.mark.parametrize("suite_timeout", [False, True])
def test_req_report_6586_orchestrator_runs_in_disposable_worktree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, suite_timeout: bool
) -> None:
    """SCENARIO-REPORT-6586-DISPOSABLE covers the complete wrapper workflow."""

    active = tmp_path / ("active-timeout" if suite_timeout else "active-red")
    active.mkdir()
    _init_repo(active)
    (active / "research-roadmap.yaml").write_text("milestone: test\n", encoding="utf-8")
    conductor = active / "scripts/research_conductor.py"
    conductor.parent.mkdir()
    conductor.write_text("# protected\n", encoding="utf-8")
    _git(active, "add", ".")
    _git(active, "commit", "-qm", "base")

    call_count = 0

    def fake_owned(
        argv: list[str],
        *,
        cwd: Path,
        env: dict[str, str],
        timeout_s: float,
        display_command: str,
        cleanup_grace_s: float = 2.0,
    ) -> dict[str, object]:
        del argv, timeout_s, cleanup_grace_s
        nonlocal call_count
        call_count += 1
        is_suite = display_command == exp.SUITE_COMMAND_TEXT
        plugin = {
            "collected_count": 2,
            "nodeids_sha256": exp.sha256_json(["a", "b"]),
            "rows": (
                [{"nodeid": "a::test_bad", "outcome": "failed", "phase": "call", "longrepr": "bad"}]
                if is_suite and not suite_timeout
                else []
            ),
            "family_summaries": [],
        }
        if not (is_suite and suite_timeout):
            Path(env[exp.PLUGIN_RECEIPT_ENV]).write_text(json.dumps(plugin), encoding="utf-8")
        return {
            "command": display_command,
            "argv": [],
            "cwd": str(cwd.resolve()),
            "exit_code": -15 if is_suite and suite_timeout else (1 if is_suite else 0),
            "timed_out": bool(is_suite and suite_timeout),
            "timeout_s": 1.0,
            "duration_s": 0.1,
            "stdout": "",
            "stderr": "",
            "process_cleanup": {
                "clean": True,
                "owned_process_group": 123,
                "signals": [],
                "surviving_owned_pids": [],
                "unrelated_process_signal_count": 0,
            },
        }

    monkeypatch.setattr(exp, "run_owned_command", fake_owned)
    report = exp.run_experiment(active, "20260825")
    assert call_count == 2
    assert report["status"] == ("timeout" if suite_timeout else "measured_red")
    assert report["disposable_checkout_receipt"]["active_root"] == str(active.resolve())
    assert report["disposable_checkout_receipt"]["cleanup"]["removed"] is True
    assert exp.validate_report(report) == []
    assert (active / exp.RESULT_RELATIVE_PATH).is_file()


def test_req_report_6586_orchestrator_writes_structured_isolation_block(tmp_path: Path) -> None:
    """REQ-REPORT-6586-VERDICT keeps a Git precondition failure recheckable."""

    active = tmp_path / "not-a-git-repo"
    active.mkdir()
    report = exp.run_experiment(active, "20260825")
    assert report["status"] == "isolated_environment_block"
    assert report["gate_check_summary"][0]["check"] == "git_command"
    assert exp.validate_report(report) == []


def test_req_report_6586_remaining_reducer_and_cleanup_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-6586-ATOMIC covers rare cleanup and reducer branches."""

    monkeypatch.setattr(exp, "_index_rows", lambda _root: {})
    monkeypatch.setattr(exp, "_run_git", lambda _root, _args: b"not-indexed\0")
    assert exp.snapshot_tracked_files(tmp_path) == {}
    assert exp._family("tests/python/test_arc_case.py::test_x") == "arc"
    assert exp._family("tests/python/test_experiment_1.py::test_x") == "experiments"
    exp.pytest_runtest_logreport(
        SimpleNamespace(passed=True, when="setup", failed=False, skipped=False)
    )

    original_read_text = Path.read_text

    def no_model(path: Path, *args: object, **kwargs: object) -> str:
        if str(path) == "/proc/cpuinfo":
            return "processor: 0\n"
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", no_model)
    assert exp._cpu_model()

    def unreadable(path: Path, *args: object, **kwargs: object) -> str:
        if str(path) == "/proc/cpuinfo":
            raise OSError("unreadable")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", unreadable)
    assert exp._cpu_model()

    ignored_child = (
        "import subprocess,sys; "
        "subprocess.Popen([sys.executable,'-c',"
        "'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)'],"
        "stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)"
    )
    leak = exp.run_owned_command(
        [sys.executable, "-c", ignored_child],
        cwd=tmp_path,
        env=os.environ.copy(),
        timeout_s=2.0,
        display_command="ignored leaked child",
        cleanup_grace_s=0.03,
    )
    assert leak["process_cleanup"]["signals"][-1]["signal"] == "SIGKILL"

    class FakeProcess:
        pid = 424242
        returncode = 0

        def communicate(self, timeout: float | None = None) -> tuple[str, str]:
            del timeout
            return "", ""

    member_rows = iter(([7], [7], [], [], []))
    monkeypatch.setattr(exp.subprocess, "Popen", lambda *_args, **_kwargs: FakeProcess())
    monkeypatch.setattr(exp, "_owned_group_members", lambda _group: list(next(member_rows)))
    monkeypatch.setattr(
        exp,
        "_signal_owned_group",
        lambda group, sig, rows: rows.append(
            {"process_group": group, "signal": sig.name, "target": "owned_process_group"}
        ),
    )
    synthetic = exp.run_owned_command(
        ["synthetic"],
        cwd=tmp_path,
        env={},
        timeout_s=1.0,
        display_command="synthetic cleanup",
        cleanup_grace_s=0.1,
    )
    assert synthetic["process_cleanup"]["leaked_owned_pids_before_cleanup"] == [7]


def test_req_report_6586_green_and_block_reports_cover_validation_edges(tmp_path: Path) -> None:
    """REQ-REPORT-6586-VERDICT validates honest GREEN and failed isolation."""

    evidence = _base_evidence(tmp_path)
    suite = deepcopy(evidence["suite"])
    suite["exit_code"] = 0
    green = exp.build_report(
        run_date="20260825",
        preconditions={},
        checkout=evidence["checkout"],
        collection=evidence["collection"],
        suite=suite,
        rows=[],
        mutation_rows=[],
        active_unchanged=evidence["active"],
        protected=evidence["protected"],
        tests_run=[],
        duration_s=1.0,
    )
    assert green["status"] == "measured_green"
    broken = deepcopy(green)
    broken["disposable_checkout_receipt"]["checkout_root"] = broken["disposable_checkout_receipt"][
        "active_root"
    ]
    broken["disposable_checkout_receipt"]["overlay_complete"] = False
    broken["suite_command_receipt"]["timed_out"] = True
    broken["reproducibility_checksum"] = exp.artifact_checksum(broken)
    errors = exp.validate_report(broken)
    assert "active_root_execution" in errors
    assert "dirty_overlay_incomplete" in errors
    assert "green_with_timeout" in errors

    checkout = deepcopy(evidence["checkout"])
    checkout["overlay_complete"] = False
    blocked = exp.build_report(
        run_date="20260825",
        preconditions={},
        checkout=checkout,
        collection=evidence["collection"],
        suite=evidence["suite"],
        rows=evidence["rows"],
        mutation_rows=[],
        active_unchanged=evidence["active"],
        protected=evidence["protected"],
        tests_run=[],
        duration_s=1.0,
    )
    assert blocked["status"] == "isolated_environment_block"
    assert "dirty_overlay_complete" in blocked["honest_verdict"]


def test_req_report_6586_atomic_error_cleanup_removes_temporary_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-6586-ATOMIC removes same-directory files after replace errors."""

    with monkeypatch.context() as patcher:
        patcher.setattr(
            exp.os, "replace", lambda _source, _target: (_ for _ in ()).throw(OSError("replace"))
        )
        with pytest.raises(OSError, match="replace"):
            exp._atomic_json(tmp_path / "plugin.json", {"ok": True})
    assert not list(tmp_path.glob(".plugin.json.*.tmp"))

    evidence = _base_evidence(tmp_path)
    report = exp.build_report(
        run_date="20260825",
        preconditions={},
        checkout=evidence["checkout"],
        collection=evidence["collection"],
        suite=evidence["suite"],
        rows=evidence["rows"],
        mutation_rows=[],
        active_unchanged=evidence["active"],
        protected=evidence["protected"],
        tests_run=[],
        duration_s=1.0,
    )
    with monkeypatch.context() as patcher:
        patcher.setattr(
            exp.os, "replace", lambda _source, _target: (_ for _ in ()).throw(OSError("replace"))
        )
        with pytest.raises(OSError, match="replace"):
            exp.atomic_write_report(tmp_path / "terminal.json", report)
    assert not list(tmp_path.glob(".terminal.json.*.tmp"))


@pytest.mark.parametrize(
    "failure",
    ["worktree_add", "revision", "overlay", "collection_timeout"],
)
def test_req_report_6586_orchestrator_isolation_failures_are_atomic_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """REQ-REPORT-6586-PRECONDITIONS preserves each setup failure as a block."""

    active = tmp_path / f"active-{failure}"
    active.mkdir()
    _init_repo(active)
    (active / "research-roadmap.yaml").write_text("milestone: test\n", encoding="utf-8")
    conductor = active / "scripts/research_conductor.py"
    conductor.parent.mkdir()
    conductor.write_text("# protected\n", encoding="utf-8")
    _git(active, "add", ".")
    _git(active, "commit", "-qm", "base")

    if failure == "worktree_add":
        original_run = exp.subprocess.run

        def fail_add(argv: list[str], **kwargs: object) -> object:
            if argv[:3] == ["git", "worktree", "add"]:
                return SimpleNamespace(returncode=1, stdout="", stderr="add failed")
            return original_run(argv, **kwargs)

        monkeypatch.setattr(exp.subprocess, "run", fail_add)
    elif failure == "revision":
        original_revision = exp.git_revision

        def wrong_revision(root: Path) -> str:
            return original_revision(root) if root == active else "b" * 40

        monkeypatch.setattr(exp, "git_revision", wrong_revision)
    elif failure == "overlay":
        monkeypatch.setattr(exp, "overlay_is_complete", lambda *_args: False)
    else:

        def timed_collection(*_args: object, **kwargs: object) -> dict[str, object]:
            return {
                "command": kwargs["display_command"],
                "cwd": str(kwargs["cwd"].resolve()),
                "timed_out": True,
                "duration_s": 0.1,
                "exit_code": -15,
                "stdout": "",
                "stderr": "",
                "process_cleanup": {
                    "clean": True,
                    "surviving_owned_pids": [],
                    "unrelated_process_signal_count": 0,
                },
            }

        monkeypatch.setattr(exp, "run_owned_command", timed_collection)

    report = exp.run_experiment(active, "20260825")
    assert report["status"] == "isolated_environment_block"
    assert exp.validate_report(report) == []

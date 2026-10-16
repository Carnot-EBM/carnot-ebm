"""REQ-REPORT-8587, REQ-VERIFY-8587: qualify real publication execution."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import coverage
import pytest

from carnot.reporting import arc_publication_8587 as e
from carnot.reporting import arc_publication_runner_8587 as run
from carnot.reporting.current_work_receipt import atomic_json, sha256_file


def cli(tmp_path, *arguments):
    """A fresh outside-checkout process must locate the shipped code itself."""
    argv = [sys.executable, "-u", str(e.old.ROOT / e.CLI), *map(str, arguments)]
    measured = coverage.Coverage.current()
    if measured is not None:
        argv[1:2] = ["-m", "coverage", "run", "--rcfile=" + str(measured.config.config_file)]
    return subprocess.run(argv, cwd=tmp_path, capture_output=True, text=True, timeout=180)


def test_authenticated_exact_historical_difference(monkeypatch):
    """REQ-REPORT-8587: original SQL and source identify only the closure call."""
    receipt = e.historical_receipt()
    assert receipt["passed"]
    assert sorted(receipt["statement_differences"].values()) == [[], [], [], [], [282]]
    assert receipt["historical_missing_statement"] == "with ("
    assert receipt["historical_validator_call"] == dict(line=265, executed=True)
    assert receipt["prompt_diagnosis_matches_authenticated_source"] is False
    assert receipt["current_statement_line"] == 280
    monkeypatch.setattr(e, "UPSTREAM_PIN", "sha256:bad")
    with pytest.raises(ValueError):
        e.historical_receipt()


@pytest.mark.parametrize("ready", [0, 1])
def test_actual_future_panel_consumer(tmp_path, ready):
    """SCENARIO-REPORT-8587-PUBLICATION: use the actual future task's gate."""
    candidate = tmp_path / "candidate.json"
    atomic_json(candidate, dict(task_id=e.TASK, arc_panel_execution_ready_score=ready))
    receipt = e.panel_consumer(candidate)
    assert receipt["accepted"] is bool(ready)
    assert receipt["matched_expectation"]
    assert receipt["consumer_task_id"] == "exp8588-arc-capability-panel"
    assert receipt["primary_sha256"] == sha256_file(candidate)


def test_private_real_publication_child(tmp_path):
    """REQ-VERIFY-8587: no success stubs may replace publication or consumers."""
    target = tmp_path / "cases.json"
    result = cli(tmp_path, "--publication-cases", target)
    assert result.returncode == 0, result.stdout + result.stderr
    receipt = json.loads(target.read_bytes())
    assert receipt["passed"]
    assert receipt["valid"]["terminal"]["panel_consumer"]["matched_expectation"]
    assert receipt["owned_failure"]["terminal"]["panel_consumer"]["matched_expectation"]
    assert receipt["invalid"]["rejected"] and receipt["invalid"]["primary_unchanged"]
    assert receipt["invalid"]["panel_consumer"]["accepted"] is False


def test_build_missing_owned_failure_and_tamper(tmp_path):
    """SCENARIO-VERIFY-8587-REPLAY: owned failure never becomes readiness."""
    with run.adapters():
        work = e.runner.empty_work(tmp_path)
        path = tmp_path / "work.json"
        output = tmp_path / (e.NAME + ".json")
        atomic_json(path, work)
        value = e.build(path, output)
        assert value["verdict_class"] == "blocked" and e.old.replay(value)
        assert value["experiment_id"] == 8587 and value["arc_panel_execution_ready_score"] == 0
        work["checks"].append(dict(name="owned_failure", passed=False))
        atomic_json(path, work)
        assert e.build(path, output)["verdict_class"] == "disqualified"
        changed = deepcopy(value)
        changed["reproducibility_checksum"] = e.old.canonical_hash(changed)
        assert not e.old.replay(changed)


def test_private_cli_and_rejections(tmp_path):
    """REQ-VERIFY-8587: real terminal replay checks immutable candidate bytes."""
    output = tmp_path / (e.NAME + ".json")
    result = cli(tmp_path, "--private-control", "--owned-failure", "--output", output)
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "disqualified"
    assert cli(tmp_path, "--cold-replay", output).returncode == 0
    value["completed_count"] = 999
    value["reproducibility_checksum"] = e.old.canonical_hash(value)
    atomic_json(output, value)
    assert cli(tmp_path, "--cold-replay", output).returncode == 1
    assert cli(tmp_path, "--cold-replay", tmp_path / "missing").returncode == 1
    assert cli(tmp_path, "--date", "20261015").returncode == 2


def test_independent_operand_and_gate_drift(tmp_path, monkeypatch):
    """REQ-VERIFY-8587: changed SQL operands and a changed future gate are rejected."""
    with monkeypatch.context() as patcher:
        patcher.setattr(e.coverage.CoverageData, "lines", lambda *_: [])
        with pytest.raises(ValueError, match="historical_statement_operands"):
            e.historical_receipt()
    with monkeypatch.context() as patcher:
        patcher.setattr(
            e.yaml,
            "safe_load",
            lambda _: dict(tasks=[dict(id="exp8588-arc-capability-panel", gated_on=[])]),
        )
        with pytest.raises(ValueError, match="future_panel_gate_drift"):
            e.panel_consumer(tmp_path / "absent")


@pytest.mark.parametrize("accepted", [None, False, True])
def test_measurement_input_and_child_failures(tmp_path, monkeypatch, accepted):
    """REQ-VERIFY-8587: a terminal failure remains distinct from source absence."""

    def history():
        if accepted is None:
            raise ValueError("absent_original")
        return dict(passed=True)

    monkeypatch.setattr(e, "historical_receipt", history)
    monkeypatch.setattr(e.old, "authenticate", lambda _: ({"historical_model_provenance": {}}, {}))
    monkeypatch.setattr(e.old, "history_receipts", lambda _: {})
    monkeypatch.setattr(e.old, "episode_hashes", lambda _: {})
    monkeypatch.setattr(e.old, "private_history", lambda *_: tmp_path / "candidate")

    def execute(name, argv, *_a, **_k):
        if name in ("primitive_replay", "original_fixtures") and accepted:
            atomic_json(Path(argv[-1]), [] if name == "primitive_replay" else {})
        return dict(name=name, passed=accepted)

    monkeypatch.setattr(e.runner, "invoke", execute)
    with run.adapters():
        work = e.runner.empty_work(tmp_path)
        run.measurement(work, e.old.ROOT, tmp_path, tmp_path)
    if accepted is None:
        assert work["failures"][0]["observed"] == "absent_original"
    else:
        assert work["terminal_replays"][0]["passed"] is accepted


@pytest.mark.parametrize("passed", [False, True])
def test_validation_keeps_publication_child_failure(tmp_path, monkeypatch, passed):
    """SCENARIO-VERIFY-8587-REPLAY: a failed publication child cannot qualify."""
    monkeypatch.setattr(run, "VALIDATION", lambda *_: None)

    def execute(name, argv, *_a, **_k):
        if passed:
            atomic_json(Path(argv[-1]), dict(passed=True))
        return dict(name=name, passed=passed)

    monkeypatch.setattr(e.runner, "invoke", execute)
    work = dict(checks=[])
    run.validation(work, tmp_path, tmp_path, tmp_path / "unused.ini")
    assert work["checks"][0]["passed"] is passed
    assert work.get("publication_cases", {}).get("passed", False) is passed


def test_replay_rejects_changed_publication_and_history(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8587-REPLAY: rehashed receipts retain independent operands."""
    monkeypatch.setattr(e, "REPLAY", lambda _: True)
    with run.adapters():
        work = e.runner.empty_work(tmp_path)
        receipt = tmp_path / "cases.json"
        atomic_json(receipt, dict(passed=False))
        work.update(
            publication_cases=dict(passed=False),
            publication_cases_reference=dict(path=str(receipt), sha256=sha256_file(receipt)),
        )
        path = tmp_path / "work.json"
        atomic_json(path, work)
        value = e.build(path, tmp_path / (e.NAME + ".json"))
        assert value["verdict_class"] == "disqualified"
        assert not e.replay(value)
        work["publication_cases"] = dict(passed=True, immutable_snapshots=[])
        atomic_json(path, work)
        assert not e.replay(e.build(path, tmp_path / (e.NAME + ".json")))
        primitive = tmp_path / "primitive.json"
        atomic_json(primitive, dict(measured=True))
        work["publication_cases"]["immutable_snapshots"] = [
            dict(path=str(primitive), sha256=sha256_file(primitive))
        ]
        atomic_json(receipt, work["publication_cases"])
        work["publication_cases_reference"]["sha256"] = sha256_file(receipt)
        atomic_json(path, work)
        assert e.replay(e.build(path, tmp_path / (e.NAME + ".json")))
        work["publication_cases"] = {}
        work["publication_history"] = dict(passed=False)
        atomic_json(path, work)
        assert not e.replay(e.build(path, tmp_path / (e.NAME + ".json")))


def test_unmodified_historical_entry_point(tmp_path):
    """REQ-VERIFY-8587: the adapter must not hide the original shipped CLI bytes."""
    argv = [sys.executable, "-u", str(e.old.ROOT / e.prior.CLI)]
    measured = coverage.Coverage.current()
    if measured is not None:
        argv[1:2] = ["-m", "coverage", "run", "--rcfile=" + str(measured.config.config_file)]
    result = subprocess.run(
        [
            *argv,
            "--suite",
            e.old.TEST,
            "--",
            *e.runner.COMMON,
            "--collect-only",
            "-q",
            str(e.old.ROOT / e.old.TEST) + "::test_union_requires_terminal_nodes",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "test_union_requires_terminal_nodes" in result.stdout

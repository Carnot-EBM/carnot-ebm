"""REQ-REPORT-8548, REQ-VERIFY-8548: terminal bytes need real child evidence."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
from types import SimpleNamespace

import coverage
import pytest

from carnot.reporting import arc_panel_replay_8548 as e
from carnot.reporting import arc_panel_replay_runner_8548 as runner
from carnot.reporting.current_work_receipt import atomic_json, sha256_file


def test_original_seals_and_private_copy(tmp_path):
    """SCENARIO-REPORT-8548-REPLAY: original quarantine survives authentication."""
    primary, work = e.authenticate(e.ROOT)
    assert primary["verdict_class"] == "disqualified"
    assert len(work["references"]) == 16
    history = e.history_receipts(primary)
    assert history["old_timeout"]["timed_out"]
    assert history["resumed_coverage_percent"] == 24
    copied = e.private_history(primary, work, tmp_path)
    assert copied.is_file()
    value = json.loads(copied.read_bytes())
    copied_work = json.loads(Path(value["work_reference"]["path"]).read_bytes())
    assert [r["primitive_directory"] for r in value["rows"]] == [
        r["directory"] for r in copied_work["references"]
    ]
    for old, new in zip(work["references"], copied_work["references"], strict=True):
        assert Path(new["directory"]).is_relative_to(tmp_path)
        assert sha256_file(Path(new["directory"]) / "steps.jsonl") == sha256_file(
            Path(old["directory"]) / "steps.jsonl"
        )
    with pytest.raises(OSError):
        e.authenticate(tmp_path / "absent")


def test_reducer_adapter_and_independent_seals(tmp_path, monkeypatch):
    """REQ-REPORT-8548: shipped reduction runs once per stored episode."""
    primary, work = e.authenticate(e.ROOT)
    calls = []
    monkeypatch.setattr(
        e.panel,
        "reduce_episode",
        lambda r: (calls.append(r) or {"episode_id": r["unit"]["episode_id"]}, [], []),
    )
    rows = e.reduce_history(work)
    assert len(rows) == len(calls) == 16
    assert all(r["status"] == "replayed" for r in rows)
    changed = deepcopy(work)
    changed["references"][0]["steps_sha256"] = "sha256:bad"
    with pytest.raises(ValueError):
        e.episode_hashes(changed)
    monkeypatch.setattr(e, "PIN", "sha256:bad")
    with pytest.raises(ValueError):
        e.authenticate(e.ROOT)


def test_union_requires_terminal_nodes(tmp_path):
    """SCENARIO-VERIFY-8548-CHILDREN: zero exit cannot hide omitted or killed nodes."""
    log = tmp_path / "nodes.stdout"
    log.write_text("tests/python/a.py::test_a PASSED [100%]\n")
    receipt = dict(
        passed=True,
        timed_out=False,
        exit_code=0,
        stdout_path=str(log),
        collected_node_ids=["tests/python/a.py::test_a"],
    )
    assert e.exact_union(["tests/python/a.py::test_a"], [receipt])["passed"]
    for update in [
        dict(timed_out=True),
        dict(exit_code=-9),
        dict(collected_node_ids=[]),
        dict(passed=False),
    ]:
        assert not e.exact_union(["tests/python/a.py::test_a"], [dict(receipt, **update)])["passed"]
    log.write_text("tests/python/a.py::test_a SKIPPED [100%]\n")
    assert not e.exact_union(["tests/python/a.py::test_a"], [receipt])["passed"]
    assert not e.exact_union([], [])["passed"]


def test_measured_coverage_rejects_missing_and_changed_sources(tmp_path, monkeypatch):
    """REQ-VERIFY-8548: coverage operands name exact source and executed statements."""
    source = tmp_path / "source.py"
    source.write_text("x = 1\ny = 2\n")
    monkeypatch.setattr(e, "OWNED", [str(source)])
    config = tmp_path / "coverage.ini"
    config.write_text("[run]\ndata_file=" + str(tmp_path / ".coverage") + "\n")
    cov = coverage.Coverage(config_file=str(config))
    cov.start()
    exec(compile(source.read_text(), str(source), "exec"), {})
    cov.stop()
    cov.save()
    shutil.copyfile(tmp_path / ".coverage", tmp_path / ".coverage.unit")
    pins = {str(source): sha256_file(source)}
    receipt = e.coverage_receipt(config, pins)
    assert receipt["passed"] and receipt["files"][str(source)]["executed_lines"] == [1, 2]
    assert not e.coverage_receipt(config, {str(source): "sha256:bad"})["passed"]
    source.write_text(source.read_text() + "z = 3\n")
    assert not e.coverage_receipt(config, {str(source): sha256_file(source)})["passed"]


def test_build_replay_and_tamper(tmp_path):
    """SCENARIO-REPORT-8548-REPLAY: outer rehashing cannot invent a qualification."""
    work = runner.empty_work(tmp_path)
    path = tmp_path / "work.json"
    atomic_json(path, work)
    output = tmp_path / (e.NAME + ".json")
    value = e.build(path, output)
    assert value["verdict_class"] == "blocked"
    assert e.replay(value)
    changed = deepcopy(value)
    changed["arc_panel_execution_ready_score"] = 1
    assert not e.replay(changed)
    work["checks"] = [dict(name="owned_failure", passed=False)]
    atomic_json(path, work)
    value = e.build(path, output)
    assert value["verdict_class"] == "disqualified" and e.replay(value)
    assert not e.replay({})
    assert e.MODEL_SPECS == []


def cli(tmp_path, *args):
    """Real standalone processes prove propagation without repository import help."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    measured = coverage.Coverage.current()
    argv = [sys.executable, "-u", str(e.ROOT / e.CLI), *map(str, args)]
    if measured is not None:
        argv[1:2] = ["-m", "coverage", "run", "--rcfile=" + str(measured.config.config_file)]
    return subprocess.run(
        argv,
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_actual_private_cli_and_consumers(tmp_path):
    """REQ-REPORT-8548: private blocked bytes and owned failures cross actual consumers."""
    for failed in [False, True]:
        output = tmp_path / str(failed) / (e.NAME + ".json")
        args = ["--private-control", "--output", output]
        if failed:
            args += ["--owned-failure"]
        result = cli(tmp_path, *args)
        assert result.returncode == 0, result.stdout + result.stderr
        value = json.loads(output.read_bytes())
        assert value["verdict_class"] == ("disqualified" if failed else "blocked")
        assert cli(tmp_path, "--cold-replay", output).returncode == 0
        side = json.loads(Path(value["terminal_validation_sidecar_path"]).read_bytes())
        assert side["passed"] and side["consumer"]["passed"]
        value["completed_count"] = 16
        atomic_json(output, value)
        assert cli(tmp_path, "--cold-replay", output).returncode == 1
    for args, expected in [
        (["--date", "20261014"], 2),
        (["--deliberate-error"], 7),
        (["--cold-replay", tmp_path / "absent"], 1),
        (["--private-control"], 2),
    ]:
        assert cli(tmp_path, *args).returncode == expected


@pytest.mark.parametrize("accepted", [False, True])
def test_measurement_keeps_real_reproduction_contract(tmp_path, monkeypatch, accepted):
    """REQ-VERIFY-8548: old kills are observations while required replay failures remain failures."""
    monkeypatch.setattr(e, "authenticate", lambda _: ({"historical_model_provenance": {}}, {}))
    monkeypatch.setattr(e, "history_receipts", lambda _: {})
    monkeypatch.setattr(e, "episode_hashes", lambda _: {})
    monkeypatch.setattr(e, "private_history", lambda *_: tmp_path / "candidate")
    raw = tmp_path / "raw"
    raw.mkdir()

    def execute(name, argv, logs, **kwargs):
        if name == "primitive_replay" and accepted:
            atomic_json(raw / "replay_rows.json", [])
        if name == "original_fixtures" and accepted:
            atomic_json(raw / "fixtures.json", {})
        if name == "partial_json":
            atomic_json(raw / "partial_coverage.json", dict(totals=dict(percent_covered=24)))
        return dict(
            name=name,
            passed=accepted,
            timed_out=name == "old90_reproduction",
            exit_code=-9 if name == "old90_reproduction" else 0,
        )

    monkeypatch.setattr(runner, "invoke", execute)
    work = runner.empty_work(tmp_path)
    runner.measurement(work, e.ROOT, raw, tmp_path)
    assert len(work["reproductions"]) == 2
    assert work["terminal_replays"][0]["name"] == "original_terminal_replay"
    assert work["checks"][-1]["passed"] is accepted


def test_validation_preserves_original_collection(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8548-CHILDREN: collect all originals before any shard executes."""
    suites = ["tests/python/a.py", "tests/python/test_8210.py"]
    monkeypatch.setattr(runner, "HISTORICAL", suites)
    calls = []

    def execute(name, argv, logs, **kwargs):
        calls.append(name)
        log = tmp_path / (name + ".stdout")
        if name.startswith("collect"):
            log.write_text(argv[-1] + "::test_original\n")
        else:
            log.write_text(
                "".join(
                    n + " PASSED [100%]\n" for n in argv if n.startswith("tests/") and "::" in n
                )
                + "1 passed\n"
            )
        return dict(name=name, passed=True, timed_out=False, exit_code=0, stdout_path=str(log))

    monkeypatch.setattr(runner, "invoke", execute)
    work = runner.empty_work(tmp_path)
    runner.validation(work, tmp_path / "raw", tmp_path, tmp_path / "coverage.ini")
    assert calls[:2] == ["collect_0", "collect_1"]
    assert work["union"]["passed"] and len(work["union"]["terminal_node_outcomes"]) == 2


@pytest.mark.parametrize("authorized", [False, True])
def test_orchestrator_resource_gate_and_qualification(tmp_path, monkeypatch, authorized):
    """REQ-REPORT-8548: owned failed checks disqualify; absent authority stays blocked."""
    (tmp_path / "research-roadmap.yaml").write_text("tasks: []\n")
    monkeypatch.setattr(e, "TASK_PIN", e.canonical_hash(None) if authorized else "bad")
    monkeypatch.setattr(
        runner, "measurement", lambda work, *_: work.update(failures=[], replay_rows=[])
    )
    monkeypatch.setattr(runner, "validation", lambda *_: None)
    monkeypatch.setattr(e, "coverage_receipt", lambda *_: dict(passed=True))
    monkeypatch.setattr(runner, "invoke", lambda name, *_a, **_k: dict(name=name, passed=True))
    monkeypatch.setattr(runner, "publish", lambda *_: dict(passed=True))
    output = tmp_path / (e.NAME + ".json")
    monkeypatch.setattr(
        runner.coverage,
        "Coverage",
        lambda **_: SimpleNamespace(start=lambda: None, stop=lambda: None, save=lambda: None),
    )
    assert runner.run(tmp_path, output, tmp_path / "scratch") == 0


def test_terminal_rebuild_and_sealed_operands(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8548-REPLAY: independently sealed history defeats rehashed summaries."""
    work = runner.empty_work(tmp_path)
    work.update(
        failures=[],
        replay_rows=[dict(episode_id=str(i)) for i in range(16)],
        checks=[dict(name=n, passed=True) for n in e.REQUIRED],
        coverage=dict(passed=True),
        union=dict(passed=True, collected_node_ids=[]),
        coverage_config=str(tmp_path / "coverage.ini"),
    )
    monkeypatch.setattr(e, "authenticate", lambda _: ({}, {}))
    monkeypatch.setattr(e, "episode_hashes", lambda _: {})
    monkeypatch.setattr(e, "reduce_history", lambda _: work["replay_rows"])
    monkeypatch.setattr(e, "history_receipts", lambda _: {})
    monkeypatch.setattr(e, "coverage_receipt", lambda *_: dict(passed=True))
    monkeypatch.setattr(e, "exact_union", lambda *_: dict(passed=True, collected_node_ids=[]))
    path = tmp_path / "work.json"
    output = tmp_path / (e.NAME + ".json")
    atomic_json(path, work)
    value = e.build(path, output)
    assert value["arc_panel_execution_ready_score"] == 1 and e.replay(value)
    monkeypatch.setattr(e, "reduce_history", lambda _: [])
    assert not e.replay(value)
    monkeypatch.setattr(e, "reduce_history", lambda _: work["replay_rows"])
    monkeypatch.setattr(e, "coverage_receipt", lambda *_: dict(passed=False))
    assert not e.replay(value)
    work["code_hashes"][str(tmp_path / "absent")] = "bad"
    atomic_json(path, work)
    assert not e.replay(e.build(path, output))


def test_cli_primitive_and_fixture_adapters(tmp_path, monkeypatch):
    """REQ-REPORT-8548: real dispatch reuses original controls without fresh episodes."""
    monkeypatch.setattr(e, "authenticate", lambda _: ({}, {}))
    monkeypatch.setattr(e, "reduce_history", lambda _: [])
    assert runner.main(["--primitives", str(tmp_path / "rows.json")]) == 0
    monkeypatch.setattr(runner.fixtures, "collect", lambda _: {})
    monkeypatch.setattr(runner.fixtures, "replay", lambda _: True)
    assert runner.main(["--fixtures", str(tmp_path / "fixture.json")]) == 0
    (tmp_path / "bad.json").write_text("{")
    assert runner.main(["--cold-replay", str(tmp_path / "bad.json")]) == 1


def test_authentication_rejects_original_authority_drift(tmp_path, monkeypatch):
    """REQ-REPORT-8548: the historical date and authority remain independently binding."""
    monkeypatch.setattr(e.panel, "TASK_PIN", "sha256:bad")
    with pytest.raises(ValueError, match="original_authority"):
        e.authenticate(e.ROOT)


def test_coverage_unions_actual_parallel_shard(tmp_path, monkeypatch):
    """REQ-VERIFY-8548: shard rows retain the executed acceptance and rejection lines."""
    source = tmp_path / "source.py"
    source.write_text("x = 1\n")
    monkeypatch.setattr(e, "OWNED", [str(source)])
    config = tmp_path / "coverage.ini"
    config.write_text("[run]\ndata_file=" + str(tmp_path / ".coverage") + "\n")
    data = coverage.CoverageData(basename=str(tmp_path / ".coverage"))
    data.add_lines({str(source): {1}})
    data.write()
    shard = coverage.CoverageData(basename=str(tmp_path / ".coverage.child"))
    shard.add_lines({str(source): {1}})
    shard.write()
    receipt = e.coverage_receipt(config, {str(source): sha256_file(source)})
    assert receipt["passed"] and receipt["shards"][0]["executed_lines"][str(source)] == [1]


def test_measurement_cap_is_owned_failure(tmp_path, monkeypatch):
    """REQ-VERIFY-8548: the measurement deadline remains a required owned failure."""
    (tmp_path / "research-roadmap.yaml").write_text("tasks: []\n")
    monkeypatch.setattr(e, "TASK_PIN", e.canonical_hash(None))
    monkeypatch.setattr(
        runner,
        "measurement",
        lambda work, *_: work.update(
            failures=[], replay_rows=[dict(observation=dict(actual_disabled_attempts=1))]
        ),
    )
    monkeypatch.setattr(runner, "validation", lambda *_: None)
    monkeypatch.setattr(e, "coverage_receipt", lambda *_: dict(passed=True))
    monkeypatch.setattr(runner, "invoke", lambda name, *_a, **_k: dict(name=name, passed=True))
    ticks = [0]

    def clock():
        ticks[0] += 1
        return 0 if ticks[0] == 1 else 4000

    monkeypatch.setattr(runner.time, "monotonic", clock)
    captured = []
    monkeypatch.setattr(
        runner, "publish", lambda value, *_: captured.append(value) or dict(passed=True)
    )
    monkeypatch.setattr(
        runner.coverage,
        "Coverage",
        lambda **_: SimpleNamespace(start=lambda: None, stop=lambda: None, save=lambda: None),
    )
    assert runner.run(tmp_path, tmp_path / (e.NAME + ".json"), tmp_path / "scratch") == 0
    assert captured[0]["verdict_class"] == "disqualified"


def test_suite_worker_uses_original_authority(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8548-AUTHORITY: immutable V733 reads cannot replace live V734 bytes."""
    suite = e.panel.qualification.TESTS[1]
    active = e.ROOT / "research-roadmap.yaml"
    current = sha256_file(active)
    observed = []
    monkeypatch.setattr(pytest, "main", lambda _: observed.append(sha256_file(active)) or 0)
    assert runner.historical_suite(suite, []) == 0
    assert observed == ["sha256:51357d7720fd1793769091b546c74a19841377969e303ba7ab9acf4f13c3c6e4"]
    assert sha256_file(active) == current
    assert runner.historical_suite(e.panel.TESTS[0], []) == 0
    assert observed[-1] == current
    monkeypatch.setattr(e.panel.qualification, "TASK_PIN", "bad")
    with pytest.raises(ValueError, match="suite_authority"):
        runner.historical_suite(suite, [])


def test_actual_original_suite_worker(tmp_path):
    """SCENARIO-VERIFY-8548-AUTHORITY: the unchanged formerly failing node passes in a real child."""
    suite = e.panel.qualification.TESTS[1]
    result = cli(
        tmp_path,
        "--suite",
        suite,
        "--",
        *runner.COMMON,
        "-vv",
        str(e.ROOT / suite) + "::test_ready_and_original_source_replay",
        "--basetemp=" + str(tmp_path / "worker"),
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "test_ready_and_original_source_replay PASSED" in result.stdout


def test_report_auto_combine_cannot_remove_evidence(tmp_path, monkeypatch):
    """REQ-VERIFY-8548: report-time shard deletion must preserve the exact original bytes."""
    (tmp_path / "research-roadmap.yaml").write_text("tasks: []\n")
    monkeypatch.setattr(runner, "measurement", lambda *_: None)
    monkeypatch.setattr(runner, "validation", lambda *_: None)
    monkeypatch.setattr(e, "coverage_receipt", lambda *_: dict(passed=True))
    monkeypatch.setattr(
        runner.coverage,
        "Coverage",
        lambda **_: SimpleNamespace(start=lambda: None, stop=lambda: None, save=lambda: None),
    )
    shard = tmp_path / "scratch" / "coverage_data" / ".coverage.real-child"

    def execute(name, *_a, **_k):
        if name == "missing":
            shard.write_bytes(b"immutable child coverage operand")
        if name == "coverage_report":
            shard.unlink()
        return dict(name=name, passed=True)

    monkeypatch.setattr(runner, "invoke", execute)
    monkeypatch.setattr(runner, "publish", lambda *_: dict(passed=True))
    assert runner.run(tmp_path, tmp_path / (e.NAME + ".json"), tmp_path / "scratch") == 0
    kept = list((tmp_path / "raw").glob("*/invocations/*/coverage_data/.coverage.real-child"))
    assert len(kept) == 1 and kept[0].read_bytes() == b"immutable child coverage operand"

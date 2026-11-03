"""REQ-REPORT-8750, REQ-VERIFY-8750: observer evidence cannot explain old timeouts."""

from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from carnot.reporting import deadline_observer_8750 as e


def test_historical_observations_are_separate():
    """SCENARIO-REPORT-8750-OBSERVER: authenticate both actual observations."""
    history, signal = e.lineage()
    assert history["observed_timeout_reference"]["receipt"]["timed_out"]
    assert signal["receipt"]["exit_code"] == -11
    assert not signal["receipt"]["timed_out"]
    assert "profile" in Path(signal["stderr"]["path"]).read_text()


def test_supervision_has_real_resource_samples_and_cleanup(tmp_path, monkeypatch):
    """REQ-VERIFY-8750: reuse repaired supervision and retain actual snapshots."""
    monkeypatch.setattr(e, "SAMPLE_S", 0.01)
    for name, code, budget in [
        ("ok", "print('8750')", 30),
        ("killed", "import time; time.sleep(20)", 0.05),
    ]:
        receipt = e.supervise(name, [sys.executable, "-c", code], tmp_path / name, budget)
        assert receipt["cleanup_passed"]
        assert receipt["timed_out"] == (name == "killed")
        assert receipt["passed"] == (name == "ok")
        samples = [
            json.loads(s) for s in Path(receipt["parent_samples"]["path"]).read_text().splitlines()
        ]
        assert samples and samples[0]["processes"]
    assert e.snapshot(999999999)["resources"] == {}


def test_unknown_cause_and_failed_owned_check_cannot_release(tmp_path):
    """REQ-REPORT-8750-PUBLICATION: diagnostic exits and validation differ."""
    work = e.old.empty_work(tmp_path, "20261103")
    work.update(observer_comparison_rows=[], historical_signal={}, repository_health_reference={})
    work["checks"] = {"owned": True}
    e.old.seal(work)
    value = e.build(work)
    assert value["verdict_class"] == "null" and value["deadline_controls_ready_score"] == 0
    assert value["MODEL_SPECS"] == [] and value["independent_count"] == 0
    work["checks"]["owned"] = False
    e.old.seal(work)
    assert e.build(work)["verdict_class"] == "disqualified"


@pytest.mark.parametrize("passed", [False, True])
def test_matched_plan_and_case_denominators(tmp_path, monkeypatch, passed):
    """SCENARIO-REPORT-8750-OBSERVER: unchanged nodes and capped arms precede results."""
    work = e.old.empty_work(tmp_path, "20261103")
    work["references"] = []
    calls = []

    def child(name, argv, raw, budget):
        calls.append((name, argv, budget))
        if passed:
            for elapsed in (3599, 3600, 3601):
                e.atomic_json(
                    raw / "deadline_cases" / f"case-{elapsed}" / "case.json",
                    dict(predicate_elapsed_s=elapsed, passed=True),
                )
        junit = next(a.split("=", 1)[1] for a in argv if a.startswith("--junitxml="))
        Path(junit).write_text("<testsuite/>")
        return dict(
            name=name,
            passed=passed,
            normal_exit=passed,
            timed_out=not passed,
            exit_code=0 if passed else -9,
            cleanup_passed=True,
            duration_s=0.1,
        )

    monkeypatch.setattr(e, "supervise", child)
    e.compare(work)
    assert len(calls) == 2 and all(c[2] == 420 for c in calls)
    assert all(c[1][-3:] == e.old.NODES for c in calls)
    assert len(work["original_deadline_rows"]) == 6
    assert len(work["observer_comparison_rows"]) == 2
    assert not work["deadline_ready"]
    assert work["changed_mechanism"] is None


def test_timer_and_stage_hooks_do_not_install_profiles(tmp_path, monkeypatch):
    """REQ-VERIFY-8750: observe outside authentication without function callbacks."""
    observed = []
    monkeypatch.setattr(
        e.faulthandler, "dump_traceback_later", lambda *a, **k: observed.append("on")
    )
    monkeypatch.setattr(
        e.faulthandler, "cancel_dump_traceback_later", lambda: observed.append("off")
    )
    e.pytest_configure(None)
    e.pytest_unconfigure(None)
    monkeypatch.setenv("CARNOT_8750_STAGES", str(tmp_path / "stages.jsonl"))
    monkeypatch.setenv("CARNOT_8750_TIMER", "1")
    e.pytest_configure(None)
    hook = e.pytest_fixture_setup(SimpleNamespace(argname="prepared"), None)
    next(hook)
    with pytest.raises(StopIteration):
        next(hook)
    e.pytest_runtest_logreport(
        SimpleNamespace(when="call", nodeid="original", outcome="passed", duration=0.1)
    )
    e.pytest_unconfigure(None)
    assert observed == ["on", "off"]
    events = [json.loads(s) for s in (tmp_path / "stages.jsonl").read_text().splitlines()]
    assert {r["stage"] for r in events} >= {"fixture_setup", "case_execution"}


def test_cli_dates_and_private_paths(tmp_path, monkeypatch):
    """REQ-REPORT-8750-PUBLICATION: reject unsafe writes before invocation work."""
    for args in (
        ["--date", "bad"],
        ["--date", "20260230"],
        ["--date", "2026113"],
        ["--private-output", str(e.ROOT / "unsafe")],
    ):
        assert e.main(args) == 2
    assert e.main(["--cold-replay", str(tmp_path / "missing")]) == 1
    monkeypatch.setattr(e, "run", lambda *args, **kwargs: 0)
    assert e.main([]) == 0


def test_private_cli_publication_and_rehashed_rejection(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8750-CUSTODY: genuine children publish and replay zero claims."""
    output = tmp_path / (e.NAME + ".json")
    command = [sys.executable, "-u", str(e.ROOT / e.CLI)]
    receipt = e.old.supervise(
        "private_cli",
        [*command, "--private-output", str(output), "--private-control"],
        tmp_path / "logs",
        90,
    )
    assert receipt["passed"], Path(receipt["stderr_path"]).read_text()
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "disqualified"
    assert e.replay(output)
    monkeypatch.setattr(e, "lineage", lambda: ({}, {}))
    assert not e.replay(output)
    monkeypatch.undo()
    for key in ("deadline_controls_ready_score", "duration_s", "diagnosed_cause"):
        changed = dict(value, **{key: "tampered"})
        changed["reproducibility_checksum"] = e.old.digest(changed)
        e.atomic_json(output, changed)
        assert not e.replay(output)
    e.atomic_json(output, value)
    work = e.old.c.read(value["primitive_reference"])
    e.atomic_json(Path(work["raw"]) / "invocation_seal.json", {})
    work["invocation_seal_reference"] = e.old.c.ref(Path(work["raw"]) / "invocation_seal.json")
    e.atomic_json(Path(work["raw"]) / "work.json", work)
    value["primitive_reference"] = e.old.c.ref(Path(work["raw"]) / "work.json")
    e.atomic_json(output, value)
    assert not e.replay(output)


def test_dispatch_and_validation_roster(tmp_path, monkeypatch):
    """REQ-VERIFY-8750: actual coverage rejects missing lines; all checks stay named."""
    output = tmp_path / (e.NAME + ".json")
    import coverage

    monkeypatch.setattr(
        coverage,
        "Coverage",
        lambda **kw: SimpleNamespace(start=lambda: None, stop=lambda: None, save=lambda: None),
    )
    monkeypatch.setattr(e, "compare", lambda work: None)
    monkeypatch.setattr(e, "validate", lambda work, meter: work["checks"].update(owned=True))
    assert e.run(output, "20261103") == 0
    assert e.replay(output)
    assert e.run(output, "20261103", private_control=True) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "disqualified"
    operand = Path(value["code_config_hashes"][0]["path"])
    ref = e.old.c.ref
    monkeypatch.setattr(
        e.old.c,
        "ref",
        lambda path: dict(path=str(path), sha256="bad") if path == operand else ref(path),
    )
    assert not e.replay(output)


def test_lineage_rejects_changed_signal_and_logs(monkeypatch):
    """REQ-VERIFY-8750: hashes and signal identity remain separate required checks."""
    original = e.c.read
    for attack in ("log", "signal"):

        def read(reference):
            value = original(reference)
            if reference == e.old.RETAINED_DIAGNOSTIC:
                value["stderr_sha256" if attack == "log" else "exit_code"] = "bad"
            return value

        with monkeypatch.context() as scope:
            scope.setattr(e.c, "read", read)
            with pytest.raises(ValueError, match="historical_signal"):
                e.lineage()


def test_replay_requires_streams_and_child_custody(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8750-CUSTODY: missing or changed raw files reject claims."""
    work = e.old.empty_work(tmp_path, "20261103")
    work.update(
        observer_comparison_rows=[],
        historical_signal={},
        repository_health_reference={},
        children=[dict(sources=[])],
    )
    path = tmp_path / "stream"
    path.write_text("original")
    work["receipts"] = [
        dict(
            name="control",
            normal_exit=True,
            passed=True,
            stdout_path=str(path),
            stderr_path=str(path),
            stdout_sha256=e.c.ref(path)["sha256"],
            stderr_sha256=e.c.ref(path)["sha256"],
        )
    ]
    e.old.seal(work)
    output = tmp_path / (e.NAME + ".json")
    e.atomic_json(output, e.build(work))
    calls = []
    monkeypatch.setattr(e, "lineage", lambda: ({}, {}))
    monkeypatch.setattr(e.c, "attest_children", lambda *a: calls.append(a))
    assert e.replay(output) and calls
    path.write_text("tampered")
    assert not e.replay(output)
    path.unlink()
    assert not e.replay(output)


def test_validation_coverage_is_real_and_health_is_separate(tmp_path, monkeypatch):
    """REQ-VERIFY-8750: a real incomplete shard cannot qualify new statements."""
    work = e.old.empty_work(tmp_path, "20261103")
    work["repository_health_reference"] = {}
    source = tmp_path / "coverage_driver.py"
    source.write_text(
        "from carnot.reporting import deadline_observer_8750 as e\ne.event('real_coverage_probe', 'end')\n"
    )
    config = tmp_path / "coverage.ini"
    config.write_text("[run]\ninclude=" + str(e.ROOT / e.OWNED[0]) + "\n")
    receipt = e.old.supervise(
        "coverage_driver",
        [
            sys.executable,
            "-m",
            "coverage",
            "run",
            "--rcfile=" + str(config),
            "--data-file=" + str(tmp_path / ".coverage.driver"),
            str(source),
        ],
        tmp_path / "logs",
        60,
    )
    assert receipt["passed"]
    assert (tmp_path / ".coverage.driver").read_bytes().startswith(b"SQLite format 3")
    meter = SimpleNamespace(stop=lambda: None, save=lambda: None)
    calls = []
    monkeypatch.setattr(e.core, "instrument", lambda *a: None)
    monkeypatch.setattr(
        e,
        "supervise",
        lambda name, argv, *a: (
            calls.append((name, argv)) or dict(name=name, passed=True, normal_exit=True)
        ),
    )
    e.validate(work, meter)
    assert not work["checks"]["new_statement_coverage"]
    assert {"e2e020", "clock_consumers", "private_e2e018", "private_e2e021"} <= {
        n for n, _ in calls
    }
    assert "full_python_suite" not in work["checks"]
    report = e.c.read(work["references"][-2])
    assert any(r["missing_lines"] for r in report["files"])

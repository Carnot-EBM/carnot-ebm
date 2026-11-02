"""REQ-REPORT-8738, REQ-VERIFY-8738: bounded diagnosis cannot rehabilitate history."""

from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from carnot.reporting import deadline_cost_8738 as e


def test_original_timeout_is_not_a_cause():
    """SCENARIO-REPORT-8738-BOUNDED: authenticate actual historical empty logs."""
    history = e.historical()
    assert history["observed_timeout_reference"]["receipt"]["exit_code"] == -9
    assert history["observed_timeout_reference"]["stdout_bytes"] == 0
    assert history["observed_timeout_reference"]["stderr_bytes"] == 0
    assert history["historical_verdict_class"] == "disqualified"
    assert sum(len(m["node_ids"]) for m in history["retained_core_manifest"]) == 39


def test_stage_reduction_requires_complete_pairs(tmp_path):
    """REQ-VERIFY-8738: active work remains censored rather than invented complete."""
    path = tmp_path / "stages.jsonl"
    assert e.stage_rows(path) == []
    events = [
        dict(event="call", stage="inspect", frame=1, monotonic_ns=10),
        dict(event="return", stage="inspect", frame=1, monotonic_ns=30),
        dict(event="call", stage="fixture_setup", frame=2, monotonic_ns=40),
    ]
    path.write_text("".join(json.dumps(v) + "\n" for v in events))
    rows = e.stage_rows(path)
    assert rows[0]["duration_s"] == 2e-8 and rows[0]["completed"]
    assert rows[1]["duration_s"] is None and not rows[1]["completed"]


def test_owned_process_timeout_and_cleanup(tmp_path):
    """SCENARIO-VERIFY-8738-CUSTODY: kill only the measured owned process tree."""
    receipt = e.supervise("normal", [sys.executable, "-c", "print('real child')"], tmp_path, 10)
    assert receipt["passed"] and receipt["normal_exit"]
    receipt = e.supervise(
        "killed", [sys.executable, "-c", "import time; time.sleep(20)"], tmp_path, 0.05
    )
    assert receipt["timed_out"] and not receipt["passed"]
    assert receipt["process_tree"] and receipt["cleanup_passed"]
    assert not Path(f"/proc/{receipt['pid']}").exists()


def test_claim_and_operand_tampering(tmp_path):
    """REQ-VERIFY-8738: the separate primitive seal defeats consistently rehashed claims."""
    raw = tmp_path / "raw"
    raw.mkdir()
    operand = raw / "operand.json"
    e.atomic_json(operand, dict(value=1))
    work = e.empty_work(raw, "20261102")
    work["references"] = [e.c.ref(operand)]
    e.seal(work)
    output = tmp_path / (e.NAME + ".json")
    value = e.build(work)
    e.atomic_json(output, value)
    assert e.replay(output)
    for key in ("deadline_controls_ready_score", "duration_s", "diagnosed_cause"):
        changed = deepcopy(value)
        changed[key] = "tampered"
        changed["reproducibility_checksum"] = e.digest(changed)
        e.atomic_json(output, changed)
        assert not e.replay(output)
    e.atomic_json(output, value)
    operand.write_text("{}")
    assert not e.replay(output)
    operand.unlink()
    assert not e.replay(output)
    assert not e.replay(tmp_path / "missing")


def test_cli_rejections(tmp_path, monkeypatch):
    """REQ-REPORT-8738: dates and private locations are checked before any write."""
    for argv in (["--date", "bad"], ["--date", "20260230"], ["--private-output", str(e.ROOT)]):
        assert e.main(argv) == 2
    assert e.main(["--cold-replay", str(tmp_path / "missing")]) == 1
    monkeypatch.setattr(e, "run", lambda *args: 0)
    assert e.main(["--date", "20261102"]) == 0


def test_historical_mutation_rejections(monkeypatch):
    """REQ-VERIFY-8738: changed receipts, source identities and custody fail explicitly."""
    read = e.c.read
    primary = read(dict(path=str(e.UPSTREAM), sha256=e.PIN))
    original = read(primary["primitive_reference"])
    for attack in ("log", "timeout", "source", "junit", "nodes", "input", "missing_sqlite"):
        work = deepcopy(original)
        smoke = next(r for r in work["receipts"] if r["name"] == "deadline_smoke")
        if attack == "log":
            smoke["stdout_sha256"] = "sha256:bad"
        elif attack == "timeout":
            smoke["exit_code"] = 0
        elif attack == "junit":
            work["children"][0]["receipt"]["junit_reference"]["sha256"] = "bad"
        elif attack == "nodes":
            work["children"][0]["node_ids"] = []
        elif attack == "input":
            work["input_references"][0]["sha256"] = "bad"
        elif attack == "missing_sqlite":
            work["children"][0]["files"][0]["reference"]["path"] = "/tmp/8738-missing-sqlite"

        def changed(reference):
            if reference == primary["primitive_reference"]:
                return work
            value = read(reference)
            if reference == work["frozen_source_bundle"] and attack == "source":
                value["sources"][0]["original"]["sha256"] = "bad"
            return value

        with monkeypatch.context() as scope:
            scope.setattr(e.c, "read", changed)
            if attack in {"junit", "nodes", "input"}:
                scope.setattr(e.core, "attest", lambda *args: {})
            if attack in {"log", "timeout"}:
                with pytest.raises(ValueError):
                    e.historical()
            else:
                assert e.historical()["retained_core_eligible_score"] == 0


def test_real_private_publication_and_replay(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8738-CUSTODY: real publication and fresh CLI replay bind zero claims."""
    import coverage

    monkeypatch.setattr(coverage, "Coverage", lambda **kwargs: SimpleNamespace(start=lambda: None))
    monkeypatch.setattr(e, "diagnostic", lambda work: None)
    monkeypatch.setattr(
        e, "validate", lambda work, driver: work["checks"].update(private_control=True)
    )
    output = tmp_path / (e.NAME + ".json")
    assert e.run(output, "20261102") == 0
    value = json.loads(output.read_bytes())
    assert value["deadline_controls_ready_score"] == 0
    assert value["verdict_class"] == "null"
    assert e.replay(output)
    work = e.c.read(value["primitive_reference"])
    work["duration_s"] += 1
    e.atomic_json(Path(value["primitive_reference"]["path"]), work)
    value["primitive_reference"] = e.c.ref(Path(value["primitive_reference"]["path"]))
    e.atomic_json(output, value)
    assert not e.replay(output)


def test_real_sqlite_and_source_fixture_hash_controls(tmp_path, monkeypatch):
    """REQ-VERIFY-8738: actual SQLite and killed receipts cannot be replaced by summaries."""
    namespace = tmp_path / "sqlite"
    namespace.mkdir()
    source = namespace / "probe.py"
    source.write_text("print('measured original control')\n")
    config = namespace / "coverage.ini"
    config.write_text(f"[run]\ndata_file={namespace / '.coverage'}\ninclude={source}\n")
    monkeypatch.setenv("COVERAGE_FILE", str(namespace / ".coverage"))
    receipt = e.supervise(
        "sqlite",
        [sys.executable, "-m", "coverage", "run", "--rcfile=" + str(config), str(source)],
        namespace,
        30,
    )
    manifest = e.c.seal_child(namespace, [e.c.ref(source)], receipt, ["probe"], "control")
    assert e.c.attest_children([manifest], manifest["sources"])
    source.write_text("print('changed source')\n")
    with pytest.raises(ValueError):
        e.c.attest_children([manifest], manifest["sources"])
    source.write_text("print('measured original control')\n")
    Path(manifest["files"][0]["reference"]["path"]).unlink()
    with pytest.raises(OSError):
        e.c.attest_children([manifest], manifest["sources"])
    killed = dict(receipt, passed=False, exit_code=-9)
    with pytest.raises(ValueError):
        e.c.seal_child(namespace, [e.c.ref(source)], killed, ["probe"], "control")


def test_profile_boundaries_and_opt_in(tmp_path, monkeypatch):
    """REQ-REPORT-8738: instrumentation observes copies and preserves real clocks."""
    path = tmp_path / "stages.jsonl"
    monkeypatch.setenv("CARNOT_8738_TRACE", str(path))
    frame = SimpleNamespace(
        f_code=SimpleNamespace(co_filename="current_work_receipt.py", co_name="atomic_json"),
        f_locals=dict(path=tmp_path / "fixture/measurement.json"),
    )
    e.profile(frame, "call", None)
    e.profile(frame, "return", None)
    assert e.stage_rows(path)[0]["stage"] == "fixture_copy:measurement.json"
    monkeypatch.setattr(e.sys, "setprofile", lambda value: None)
    monkeypatch.setattr(e.faulthandler, "dump_traceback_later", lambda *a, **k: None)
    monkeypatch.setattr(e.faulthandler, "cancel_dump_traceback_later", lambda: None)
    e.pytest_configure(None)
    e.pytest_unconfigure(None)
    monkeypatch.delenv("CARNOT_8738_TRACE")
    e.pytest_configure(None)
    e.pytest_unconfigure(None)


@pytest.mark.parametrize("passed,timed_out", [(False, True), (False, False), (True, False)])
def test_diagnostic_plan_and_terminal_accounting(tmp_path, monkeypatch, passed, timed_out):
    """REQ-REPORT-8738: preserve the frozen nodes and never retry a censored diagnostic."""
    work = e.empty_work(tmp_path, "20261102")
    monkeypatch.setattr(e, "RETAINED_DIAGNOSTIC", None)
    calls = []

    def observed(name, argv, raw, budget):
        calls.append((argv, budget))
        if passed:
            for elapsed in (3599, 3600, 3601):
                e.atomic_json(
                    raw / "deadline_cases" / f"case-{elapsed}" / "case.json", dict(passed=True)
                )
        return dict(name=name, passed=passed, duration_s=0.1, timed_out=timed_out)

    monkeypatch.setattr(e, "supervise", observed)
    monkeypatch.setattr(e.c, "seal_child", lambda *args: dict(unit_test_only=True))
    e.diagnostic(work)
    assert len(calls) == 1 and calls[0][1] == 420
    assert calls[0][0][-3:] == e.NODES
    assert len(work["original_deadline_rows"]) == 3
    assert work["deadline_ready"] is passed


def test_cleanup_handles_already_exited_child(tmp_path, monkeypatch):
    """REQ-VERIFY-8738: a child exiting during termination must still be reaped."""
    kill = e.os.kill

    def raced(pid, sig):
        kill(pid, sig)
        raise ProcessLookupError

    monkeypatch.setattr(e.os, "kill", raced)
    row = e.supervise("race", [sys.executable, "-c", "import time; time.sleep(10)"], tmp_path, 0.05)
    assert row["timed_out"] and row["cleanup_passed"]


def test_validation_uses_real_coverage_and_never_grants_missing_lines(tmp_path, monkeypatch):
    """REQ-VERIFY-8738: actual incomplete SQLite cannot qualify statement coverage."""
    work = e.empty_work(tmp_path, "20261102")
    source = tmp_path / "coverage_driver.py"
    source.write_text(
        "from carnot.reporting import deadline_cost_8738 as e\ne.progress('real driver')\n"
    )
    config = tmp_path / "coverage.ini"
    config.write_text("[run]\ninclude=" + str(e.ROOT / e.OWNED[0]) + "\n")
    receipt = e.supervise(
        "driver",
        [
            sys.executable,
            "-m",
            "coverage",
            "run",
            "--rcfile=" + str(config),
            "--data-file=" + str(tmp_path / ".coverage.driver"),
            str(source),
        ],
        tmp_path,
        30,
    )
    assert receipt["passed"]
    driver = SimpleNamespace(stop=lambda: None, save=lambda: None)
    monkeypatch.setattr(e.core, "instrument", lambda *args: None)
    monkeypatch.setattr(
        e.c,
        "supervise",
        lambda name, *args, **kwargs: dict(name=name, passed=True, normal_exit=True),
    )
    monkeypatch.setattr(
        e, "supervise", lambda name, *args: dict(name=name, passed=True, normal_exit=True)
    )
    e.validate(work, driver)
    assert work["checks"]["new_statement_coverage"] is False
    assert all(Path(r["path"]).exists() for r in work["references"])


@pytest.mark.parametrize("inherited_path", [False, True])
def test_external_driver_receives_checkout_imports(tmp_path, monkeypatch, inherited_path):
    """SCENARIO-VERIFY-8738-CHILD-IMPORTS: real children preserve both import roots."""
    extra = tmp_path / "inherited imports"
    extra.mkdir()
    (extra / "inherited_control.py").write_text("VALUE = 8738\n")
    if inherited_path:
        monkeypatch.setenv("PYTHONPATH", str(extra))
    else:
        monkeypatch.delenv("PYTHONPATH", raising=False)
    source = tmp_path / "external_driver.py"
    source.write_text(
        "from carnot.reporting import deadline_cost_8738 as e\n"
        "from scripts import conductor_gates\n"
        "print(e.__file__, conductor_gates.__file__)\n"
        + ("from inherited_control import VALUE\nassert VALUE == 8738\n" if inherited_path else "")
    )
    receipt = e.supervise("external_imports", [sys.executable, str(source)], tmp_path, 30)
    assert receipt["passed"], Path(receipt["stderr_path"]).read_text()
    assert receipt["cleanup_passed"]
    output = Path(receipt["stdout_path"]).read_text()
    assert str(e.ROOT / e.OWNED[0]) in output
    assert str(e.ROOT / "scripts/conductor_gates.py") in output


def test_replay_checks_child_and_changed_history(tmp_path, monkeypatch):
    """REQ-VERIFY-8738: cold reconstruction rechecks producer custody and old operands."""
    work = e.empty_work(tmp_path, "20261102")
    work["children"] = [dict(sources=[])]
    work["history"] = dict(old="changed")
    e.seal(work)
    output = tmp_path / (e.NAME + ".json")
    e.atomic_json(output, e.build(work))
    seen = []
    monkeypatch.setattr(e.c, "attest_children", lambda *args: seen.append(args) or {})
    assert not e.replay(output)
    assert len(seen) >= 1


def test_retained_terminal_diagnostic_never_reexecutes(tmp_path, monkeypatch):
    """REQ-VERIFY-8738: failed diagnostic evidence is terminal and its observer stays byte-bound."""
    monkeypatch.setattr(e, "supervise", lambda *args: pytest.fail("terminal evidence cannot retry"))
    work = e.empty_work(tmp_path, "20261102")
    e.diagnostic(work)
    assert not work["deadline_ready"]
    assert work["receipts"][0]["exit_code"] == -11
    assert not work["receipts"][0]["timed_out"]
    assert "signal_11" in work["diagnosed_cause"]
    assert len(work["original_deadline_rows"]) == 3
    read = e.c.read

    def changed(reference):
        value = read(reference)
        if reference["path"].endswith("diagnostic_source_snapshot/source-seal.json"):
            value["sources"][0]["frozen"]["sha256"] = "bad"
        return value

    with monkeypatch.context() as scope:
        scope.setattr(e.c, "read", changed)
        with pytest.raises(ValueError, match="retained_observer_source_hash"):
            e.diagnostic(e.empty_work(tmp_path, "20261102"))
    receipt = tmp_path / "diagnostic/receipt.json"
    e.atomic_json(receipt, work["receipts"][0])
    e.atomic_json(tmp_path / "frozen_method.json", dict(nodes=[], diagnostic_cap_s=420))
    monkeypatch.setattr(e, "RETAINED_DIAGNOSTIC", e.c.ref(receipt))
    with pytest.raises(ValueError, match="retained_diagnostic_method"):
        e.diagnostic(e.empty_work(tmp_path, "20261102"))

"""REQ-REPORT-8554, REQ-VERIFY-8554: measured nodes retain original authority."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from carnot.reporting import historical_learning_8554 as e
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.v709_execution import child


def test_partition_and_authenticated_original():
    """SCENARIO-VERIFY-8554-PARTITION: no original test can disappear or repeat."""
    old, work = e.original()
    assert old["completed_count"] == 152
    assert len(work["nodes"]) == 158
    groups = e.partition(work["nodes"])
    assert sorted(n for g in groups for n in g) == sorted(work["nodes"])
    for group in groups:
        assert len({n.split("::")[0] for n in group}) == 1
        if "8434" in group[0]:
            assert len(group) <= 4
            if any("process_exit" in n or "retention" in n for n in group):
                assert len(group) == 1
    assert work["authorities"] == e.old.authorities(work["nodes"])
    assert (
        work["authorities"]["tests/python/test_objective_reporting_8461.py"]["run_date"]
        == "20261012"
    )
    with pytest.raises(ValueError, match="duplicate"):
        e.partition(["suite::test", "suite::test"])


def test_real_child_and_killed_rejection(tmp_path):
    """SCENARIO-VERIFY-8554-PARTITION: authentic logs still require a live bounded exit."""
    node = "tests/python/test_objective_checkpoint_8461.py::test_transition_controls_and_invalid_configuration"
    receipt = e.node_child("real", [node], tmp_path / "missing" / "scratch", tmp_path / "logs")
    assert receipt["passed"]
    assert receipt["node_elapsed_s"][node] >= 0
    assert e.exact_union([node], [receipt])["passed"]
    for field, value in [("timed_out", True), ("actual_exit", -9), ("authority", {})]:
        changed = deepcopy(receipt)
        changed[field] = value
        assert not e.exact_union([node], [changed])["passed"]
    assert not e.exact_union([node, node], [receipt])["passed"]
    assert not e.exact_union([node], [receipt, receipt])["passed"]
    assert not e.exact_union([], [])["passed"]
    changed = deepcopy(receipt)
    changed["node_elapsed_s"][node] += 999
    assert not e.exact_union([node], [changed])["passed"]
    killed = e.node_child("killed", [node], tmp_path / "killed", tmp_path / "kill_logs", 0.01)
    assert killed["timed_out"] and killed["actual_exit"] < 0
    assert not e.exact_union([node], [killed])["passed"]


def test_actual_private_cli_and_tampering(tmp_path):
    """SCENARIO-REPORT-8554-REPLAY: actual publication and negative controls preserve failure."""
    output = tmp_path / "results" / (e.NAME + ".json")
    r = child(
        "private",
        [sys.executable, "-u", str(e.ROOT / e.CLI), "--private-control", "--output", str(output)],
        tmp_path / "cli",
        deadline=120,
    )
    assert r["passed"], Path(r["stderr_path"]).read_text()
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "disqualified"
    assert not value["historical_learning_validation_ready_score"]
    assert e.replay(output)
    terminal = json.loads(Path(value["terminal_validation_sidecar_path"]).read_bytes())
    assert terminal["consumer"]["passed"]
    assert terminal["primary_sha256"] == e.old.reference(output)["sha256"]
    value["historical_learning_validation_ready_score"] = 1
    value["reproducibility_checksum"] = e.report.checksum(value)
    atomic_json(output, value)
    assert not e.replay(output)
    assert not e.replay(tmp_path / "absent")
    baseline = json.loads(Path(value["primitive_reference"]["path"]).read_bytes())
    for mode in ("log", "receipt", "source", "nodes", "junit"):
        work = deepcopy(baseline)
        if mode == "log":
            work["checks"][0]["stdout_sha256"] = "wrong"
        elif mode == "receipt":
            work["checks"][0]["duration_s"] += 1
        elif mode == "nodes":
            work["nodes"] = ["foreign::test"]
        elif mode == "junit":
            work["checks"][0]["junit_reference"] = dict(path=str(output), sha256="wrong")
        raw = Path(value["primitive_reference"]["path"]).parent
        atomic_json(raw / "measurement.json", work)
        modified = e.build(work, raw, output)
        if mode == "source":
            modified["code_config_hashes"][0]["sha256"] = "wrong"
        modified["reproducibility_checksum"] = e.report.checksum(modified)
        atomic_json(output, modified)
        assert not e.replay(output)


def test_original_authentication_controls(monkeypatch):
    """SCENARIO-REPORT-8554-REPLAY: repaired manifests cannot replace pinned original source bytes."""
    original = e.old.reference
    log = Path(e.original()[1]["receipts"][0]["stdout_path"])
    with monkeypatch.context() as scoped:
        scoped.setattr(e.old, "reference", lambda p: dict(original(p), sha256="wrong"))
        with pytest.raises(ValueError, match="original_source_hash"):
            e.original()
    with monkeypatch.context() as scoped:
        scoped.setattr(
            e.old,
            "reference",
            lambda p: dict(original(p), sha256="wrong") if p == log else original(p),
        )
        with pytest.raises(ValueError, match="original_log_hash"):
            e.original()
    actual = e.old.original_union
    monkeypatch.setattr(e.old, "original_union", lambda: (actual()[0], ["foreign"]))
    with pytest.raises(ValueError, match="original_union_authority"):
        e.original()


def test_cli_rejections(tmp_path):
    """SCENARIO-REPORT-8554-REPLAY: wrong dates and public controls fail closed."""
    from carnot.reporting import historical_learning_runner_8554 as r

    assert r.main(["--deliberate-error"]) == 7
    assert r.main(["--cold-replay", str(tmp_path / "absent")]) == 1
    with pytest.raises(SystemExit):
        r.main(["--date", "20261014"])
    with pytest.raises(SystemExit):
        r.main(["--private-control"])


def test_measurement_budget_and_real_timeout(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8554-PARTITION: exhausted budgets and real kills stop scheduling immediately."""
    import time
    from carnot.reporting import historical_learning_runner_8554 as r

    original = e.original()[1]
    monkeypatch.setattr(
        r,
        "collect",
        lambda raw, suite: ([n for n in original["nodes"] if n.split("::")[0] == suite], {}),
    )
    failed = child("failure", [sys.executable, "-c", "raise SystemExit(7)"], tmp_path / "failed")
    passed = child("pass", [sys.executable, "-c", "pass"], tmp_path / "passed")
    killed = child(
        "kill",
        [sys.executable, "-c", "import time;time.sleep(10)"],
        tmp_path / "killed",
        deadline=0.01,
    )
    for mode in ("failed", "isolated_timeout", "expired", "shard_timeout"):
        work = dict(
            checks=[],
            receipts=[],
            failures=[],
            diagnostics=[],
            start_ns=time.monotonic_ns() if mode == "shard_timeout" else 0,
        )

        def node(*args, **kwargs):
            if mode == "failed":
                return failed
            if mode == "isolated_timeout":
                return killed
            if args[0] == "reproduce_node":
                return passed
            return dict(killed, node_elapsed_s={})

        monkeypatch.setattr(e, "node_child", node)
        r.measure(work, tmp_path / mode, tmp_path / "private")
        if mode == "expired":
            assert work["failures"][0]["check"] == "union_budget_s"
            assert not work["receipts"]
        else:
            assert work["diagnostics"] and not work["diagnostics"][0]["assertion_failure_inferred"]
    work.update(preconditions=dict(scratch_path=str(tmp_path)))
    fallbacks = r.fallback_suites(work)
    assert {n for n, group in fallbacks} == {"private_E2E018", "private_E2E021"}
    assert all(n in r.commands(work, tmp_path) for n, group in fallbacks)
    r.validate_consumers(work, tmp_path, tmp_path)
    work["receipts"] = [dict(passed=True, collected_node_ids=group) for n, group in fallbacks]
    assert not r.fallback_suites(work)


def test_missing_input_and_resource_gate(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8554-REPLAY: unavailable inputs and disk memory gates remain explicit."""
    from carnot.reporting import historical_learning_runner_8554 as r

    def absent(work, raw, private):
        raise FileNotFoundError("missing original bytes")

    monkeypatch.setattr(r, "measure", absent)
    invoke = r.shipped.invoke

    def fast(name, argv, raw, **kwargs):
        if name == "coverage_json":
            import coverage

            current = coverage.Coverage.current()
            if current is not None:
                current.json_report(outfile=argv[-1])
        if name == "owned_tests" or name.startswith(
            ("coverage_", "ruff_", "strict_mypy", "spec_coverage")
        ):
            return child(name, [sys.executable, "-c", "raise SystemExit(7)"], raw)
        return invoke(name, argv, raw, **kwargs)

    monkeypatch.setattr(r.shipped, "invoke", fast)
    output = tmp_path / "absent" / (e.NAME + ".json")
    assert r.run(output, tmp_path / "scratch") == 0
    assert json.loads(output.read_bytes())["gate_check_summary"][0]["field"] == "original_inputs"
    resources = r.shipped.resources
    monkeypatch.setattr(r.shipped, "resources", lambda p: dict(resources(p), filesystem="tmpfs"))
    output = tmp_path / "disk" / (e.NAME + ".json")
    assert r.run(output, tmp_path / "disk_scratch", control=True) == 0
    assert json.loads(output.read_bytes())["verdict_class"] == "blocked"


def test_standalone_coverage_bootstrap(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8554-PARTITION: standalone tracing starts a real recorder and restores its caller."""
    import coverage
    from carnot.reporting import historical_learning_runner_8554 as r

    outer = coverage.Coverage.current()
    output = tmp_path / "bootstrap" / (e.NAME + ".json")
    with monkeypatch.context() as scoped:
        scoped.setattr(coverage.Coverage, "current", classmethod(lambda cls: None))
        assert r.run(output, tmp_path / "bootstrap_scratch", control=True) == 0
    nested = coverage.Coverage.current()
    assert nested is not None and nested is not outer
    nested.stop()
    nested.save()
    assert coverage.Coverage.current() is outer
    assert e.replay(output)

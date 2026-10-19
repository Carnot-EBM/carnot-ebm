"""REQ-VERIFY-8654: private publication and cold replay use the actual CLI."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from carnot.reporting import learning_core_8654 as e
from carnot.reporting import learning_core_report_8654 as report
from carnot.reporting import learning_core_runner_8654 as runner


def test_private_cli_and_replay(tmp_path):
    """SCENARIO-VERIFY-8654-REPLAY: missing external authority stays terminal blocked."""
    output = tmp_path / (e.NAME + ".json")
    command = [sys.executable, "-u", str(e.ROOT / e.CLI)]
    receipt = e.c.supervise(
        "private",
        command
        + [
            "--date",
            "20261019",
            "--private-output",
            str(output),
            "--upstream",
            str(tmp_path / "missing"),
            "--private-control",
        ],
        tmp_path / "logs",
        deadline=120,
    )
    assert receipt["passed"], Path(receipt["stderr_path"]).read_text()
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked"
    assert value["learning_core_ready_score"] == 0
    assert value["MODEL_SPECS"] == []
    assert runner.replay(output)
    assert runner.main(["--cold-replay", str(output)]) == 0
    for mode, arguments, expected in [
        ("invalid", ["--date", "bad"], 2),
        ("calendar", ["--date", "20261020"], 2),
        ("error", ["--deliberate-error"], 7),
        ("unsafe", ["--private-control"], 2),
        ("repo", ["--private-output", str(e.ROOT / "results" / "unsafe.json")], 2),
    ]:
        assert e.c.supervise(
            mode, command + arguments, tmp_path / "logs", expected=expected, deadline=30
        )["passed"]
    for field in ("learning_core_ready_score", "duration_s", "completed_count"):
        changed = deepcopy(value)
        changed[field] += 1
        e.atomic_json(output, changed)
        assert not runner.replay(output)
    e.atomic_json(output, value)
    work = e.c.read(value["primitive_reference"])
    work["duration_s"] += 1
    e.atomic_json(Path(value["primitive_reference"]["path"]), work)
    value["primitive_reference"] = e.c.ref(Path(value["primitive_reference"]["path"]))
    e.atomic_json(output, value)
    assert not runner.replay(output)
    assert not runner.replay(tmp_path / "missing")


def test_report_gates(tmp_path):
    """REQ-REPORT-8654: owned failures disqualify; only a complete core earns readiness."""
    work = e.prepare(tmp_path / "raw", "20261019", tmp_path / "missing")
    e.atomic_json(Path(work["raw"]) / "work.json", work)
    work["invocation_seal_reference"] = dict(path="unit-seal", sha256="fixture")
    work["input_failures"] = []
    work["partition"] = dict(
        cohorts=[
            dict(cohort="A", node_ids=["original.py::test"]),
            dict(cohort="B", node_ids=["later.py::test"]),
        ],
        original_node_ids=["original.py::test", "later.py::test"],
    )
    assert report.build(work, tmp_path / (e.NAME + ".json"))["verdict_class"] == "disqualified"
    work.update(
        outcomes=[dict(node_id="original.py::test", outcome="passed")],
        children=[dict(receipt={}, files=[])],
        original_coverage=dict(observed_statements=1),
        owned_coverage=dict(complete=True),
        launch_budget_rows=[dict(admitted=True)],
        receipts=[dict(name=n, passed=True) for n in report.REQUIRED],
    )
    value = report.build(work, tmp_path / (e.NAME + ".json"))
    assert value["learning_core_ready_score"] == 1
    assert value["verdict_class"] == "circular_positive"
    assert value["scope"] == "core_only" and not value["training_eligible"]
    assert "learning_execution_ready_score" not in value
    assert value["cohort_b_pending_nodes"] == ["later.py::test"]
    work["receipts"][0]["passed"] = False
    assert report.build(work, tmp_path / (e.NAME + ".json"))["verdict_class"] == "disqualified"


def test_instrument_failure_and_real_shard(tmp_path):
    """SCENARIO-VERIFY-8654-REPLAY: killed or empty children cannot gain receipts."""
    work = e.prepare(tmp_path / "raw", "20261019", tmp_path / "missing")
    node = "tests/python/test_learning_core_8654.py::test_launch_budget"
    source = e.c.ref(e.ROOT / e.OWNED[0])
    work["mechanics_fixture"] = e.c.read(work["partition"]["historical_primitive_reference"])[
        "mechanics_fixture"
    ]
    receipt = e.instrument(work, "tiny", [e.TESTS[0]], [source], 120, "owned")
    assert receipt["passed"]
    assert work["children"] and receipt["junit_reference"]
    assert node in work["children"][0]["node_ids"]
    assert e.TESTS[0] not in work["children"][0]["node_ids"]
    failed = e.instrument(work, "killed", [node], [source], 0.001, "owned")
    assert failed["timed_out"] and failed["exit_code"] == -9
    assert work["child_errors"]
    work["sources"] = [source]
    e.recompute(work, tmp_path / "combine")
    manifest = work["children"][0]
    manifest["node_ids"] = ["omitted.py::test"]
    with pytest.raises(ValueError, match="node_assignment"):
        e.recompute(work, tmp_path / "bad")


def test_source_closure(tmp_path):
    """REQ-VERIFY-8654: include direct imports and skip absent optional local modules."""
    path = tmp_path / "local.py"
    path.write_text(
        'import os\nfrom carnot.reporting import current_work_receipt\nNAME="experiment_does_not_exist"\n'
    )
    refs = e.source_closure([path, path, tmp_path / "missing"])
    assert e.c.ref(path) in refs
    assert any(r["path"].endswith("current_work_receipt.py") for r in refs)


def test_main_dispatch(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8654-REPLAY: production uses the exact dated output path."""
    calls = []
    monkeypatch.setattr(runner, "run", lambda *a, **k: calls.append((a, k)) or 0)
    assert runner.main([]) == 0
    assert calls[0][0][0] == e.ROOT / "results" / (e.NAME + ".json")
    assert runner.main(["--private-output", str(tmp_path / (e.NAME + ".json"))]) == 0


def test_replay_receipts_and_validation_dispatch(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8654-REPLAY: actual receipt bytes cannot be repaired by summaries."""
    work = e.prepare(tmp_path / "raw", "20261019", tmp_path / "missing")
    calls = []
    monkeypatch.setattr(e, "instrument", lambda *a: calls.append(a))
    monkeypatch.setattr(e.c, "supervise", lambda name, *a, **k: dict(name=name, passed=True))
    runner.validation(work)
    assert len(calls) == 1 and len(work["receipts"]) == 6
    assert {r["name"] for r in work["receipts"]} == report.REQUIRED - {"owned_tests"}
    monkeypatch.undo()
    log = tmp_path / "log"
    log.write_text("original")
    work["receipts"] = [
        dict(
            stdout_path=str(log),
            stdout_sha256=e.c.ref(log)["sha256"],
            stderr_path=str(log),
            stderr_sha256=e.c.ref(log)["sha256"],
            passed=True,
            name="unit",
        )
    ]
    e.recompute(work, tmp_path / "copies")
    runner.freeze(work)
    output = tmp_path / (e.NAME + ".json")
    e.atomic_json(output, report.build(work, output))
    assert runner.replay(output)
    log.write_text("changed")
    assert not runner.replay(output)


def test_normal_run_dispatch_without_recursive_validation(tmp_path, monkeypatch):
    """REQ-VERIFY-8654: production calls measurement and validation before publication."""
    work = e.prepare(tmp_path / "raw", "20261019", tmp_path / "missing")
    work["input_failures"] = []
    calls = []
    monkeypatch.setattr(e, "prepare", lambda *a: work)
    monkeypatch.setattr(e, "execute", lambda *a: calls.append("execute"))
    monkeypatch.setattr(runner, "validation", lambda *a: calls.append("validate"))
    output = tmp_path / (e.NAME + ".json")
    assert runner.run(output, "20261019") == 0
    assert calls == ["execute", "validate"]
    assert runner.replay(output)


def test_recompute_rejections(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8654-REPLAY: source, JUnit, collection and combine claims stay bound."""
    work = e.prepare(tmp_path / "raw", "20261019", tmp_path / "missing")
    node = "tests/python/test_learning_core_8654.py::test_launch_budget"
    source = e.c.ref(e.ROOT / e.OWNED[0])
    assert e.instrument(work, "unitA", [node], [source], 60, "A")["passed"]
    work["sources"] = [source]
    e.recompute(work, tmp_path / "good")
    assert work["outcomes"][0]["node_id"] == node
    for mode in ("source", "config", "junit", "node", "collection", "combine"):
        changed = deepcopy(work)
        if mode == "source":
            changed["sources"][0]["sha256"] = "wrong"
        elif mode == "config":
            changed["children"][0]["receipt"]["coverage_config_reference"]["sha256"] = "wrong"
        elif mode == "junit":
            changed["children"][0]["receipt"]["junit_reference"]["sha256"] = "wrong"
        elif mode == "node":
            changed["children"][0]["node_ids"] = ["omitted::test"]
        elif mode == "collection":
            path = tmp_path / "wrong_collection.json"
            e.atomic_json(path, dict(nodes=["foreign::test"]))
            changed["collection_reference"] = e.c.ref(path)
        else:
            monkeypatch.setattr(
                e.c, "combine_copies", lambda *a: dict(second_combine_invariance=dict(passed=False))
            )
        with pytest.raises(ValueError):
            e.recompute(changed, tmp_path / mode)
        monkeypatch.undo()
    path = tmp_path / "collection.json"
    work["collected"] = work["partition"]["original_node_ids"]
    e.atomic_json(path, dict(nodes=work["collected"]))
    work["collection_reference"] = e.c.ref(path)
    e.recompute(work, tmp_path / "collected")


def test_direct_cli_rejections():
    """SCENARIO-VERIFY-8654-REPLAY: invalid calendar and unsafe controls exit before work."""
    assert runner.main(["--date", "bad"]) == 2
    assert runner.main(["--date", "20261020"]) == 2
    assert runner.main(["--private-control"]) == 2
    assert runner.main(["--deliberate-error"]) == 7

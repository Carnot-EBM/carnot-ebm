"""REQ-VERIFY-8566: actual private CLI, publication and cold rejection controls."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import count_semantics_8566 as e
from carnot.reporting import count_semantics_report_8566 as report
from carnot.reporting import count_semantics_runner_8566 as runner
from carnot.reporting.current_work_receipt import atomic_json

CLI = e.ROOT / "scripts/experiments" / (e.NAME + ".py")


def cli(*args):
    """Run a real bounded child so coverage includes direct script entry."""
    return subprocess.run(
        [sys.executable, "-u", str(CLI), *map(str, args)],
        capture_output=True,
        text=True,
        timeout=90,
    )


def test_private_cli_replay_and_mutations(tmp_path):
    """SCENARIO-VERIFY-8566-REPLAY: private publication never claims production checks."""
    output = tmp_path / (e.NAME + ".json")
    done = cli("--date", "20261016", "--fixture-output", output)
    assert done.returncode == 0, done.stdout + done.stderr
    original = json.loads(output.read_bytes())
    assert original["verdict_class"] == "disqualified"
    assert original["count_controls_ready_score"] == 0
    assert cli("--cold-replay", output).returncode == 0
    for key, value in [
        ("count_controls_ready_score", 1),
        ("expected_classification_rows", []),
        ("owned_controller_failures", []),
        ("rows", []),
        ("required_checks_passed", True),
    ]:
        changed = dict(original, **{key: value})
        changed["reproducibility_checksum"] = report.checksum(changed)
        atomic_json(output, changed)
        assert cli("--cold-replay", output).returncode == 1
    atomic_json(output, original)
    work_path = Path(original["primitive_reference"]["path"])
    work = json.loads(work_path.read_bytes())
    changed = deepcopy(work)
    changed["corpus"]["cases"][0]["left"]["value"] = 2399
    atomic_json(work_path, changed)
    changed_artifact = report.build(changed, output)
    atomic_json(output, changed_artifact)
    assert cli("--cold-replay", output).returncode == 1
    atomic_json(work_path, work)
    atomic_json(output, original)
    work_path.unlink()
    assert cli("--cold-replay", output).returncode == 1
    assert cli("--cold-replay", tmp_path / "absent.json").returncode == 1
    assert cli("--date", "20261015").returncode == 2
    assert cli("--fixture-output", e.ROOT / "results" / (e.NAME + ".json")).returncode == 2


def test_report_owned_failure_and_absent_evidence(tmp_path):
    """REQ-VERIFY-8566: failed owned work disqualifies; missing original evidence blocks."""
    work = runner.prepare(e.ROOT, tmp_path / "raw")
    output = tmp_path / (e.NAME + ".json")
    value = report.build(work, output)
    assert value["verdict_class"] == "disqualified"
    work["receipts"] = [dict(name="owned_failed", passed=False)]
    assert report.build(work, output)["count_controls_ready_score"] == 0
    work["receipts"] = []
    work["evidence"] = e.measure(tmp_path / "absent", tmp_path / "blocked")
    assert report.build(work, output)["verdict_class"] == "blocked"
    work["receipts"] = [dict(name="owned_failed", passed=False)]
    assert report.build(work, output)["verdict_class"] == "disqualified"
    assert runner.main(["--date", "20261015"]) == 2


def test_coverage_and_union_rejection(tmp_path):
    """REQ-VERIFY-8566: missing statements and duplicate node outcomes fail closed."""
    assert not runner.coverage_complete(dict(files={}))
    assert not runner.coverage_complete(dict(files={"a": dict(summary=dict(missing_lines=1))}))
    assert runner.coverage_complete(dict(files={"a": dict(summary=dict(missing_lines=0))}))
    assert runner.exact_union(["a", "b"], [["a"], ["b"]])
    assert not runner.exact_union(["a"], [["a"], ["a"]])
    assert not runner.exact_union(["a", "b"], [["a"]])


def test_terminal_retains_critical_findings(tmp_path, monkeypatch):
    """REQ-VERIFY-8566: a real critical flag cannot be masked by a missing summary key."""
    monkeypatch.setattr(
        runner, "verify_artifact", lambda p: dict(flags=[dict(severity="critical")])
    )
    monkeypatch.setattr(runner.report, "replay", lambda p: True)
    monkeypatch.setattr(runner, "child", lambda *a, **k: dict(passed=True))
    assert not runner.terminal(tmp_path / "candidate.json", tmp_path)["passed"]


def test_replay_rejects_rehashed_primitive_proofs(tmp_path):
    """REQ-VERIFY-8566: even a repaired envelope cannot alter a primitive proof."""
    work = runner.prepare(e.ROOT, tmp_path / "raw")
    output = tmp_path / (e.NAME + ".json")
    work_path = tmp_path / "raw/work.json"
    for field in ["metrics", "cardinality_reductions", "historical_controls"]:
        changed = deepcopy(work)
        if field == "metrics":
            changed["evidence"][field]["completed_count"] = 2399
        elif field == "cardinality_reductions":
            changed["evidence"][field][0]["left_value"] = 2399
        else:
            changed["evidence"][field][0]["current_critical_findings"] = []
        atomic_json(work_path, changed)
        atomic_json(output, report.build(changed, output))
        assert not report.replay(output)
    atomic_json(work_path, work)
    value = report.build(work, output)
    value["reproducibility_checksum"] = "wrong"
    atomic_json(output, value)
    assert not report.replay(output)


def test_command_manifest_and_production_route(tmp_path, monkeypatch):
    """REQ-VERIFY-8566: orchestration binds required commands and records their union."""
    work = runner.prepare(e.ROOT, tmp_path / "raw")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    called = []

    def controlled_child(name, argv, logs, **kwargs):
        called.append(name)
        logs.mkdir(parents=True, exist_ok=True)
        out, err = logs / (name + ".stdout"), logs / (name + ".stderr")
        out.write_text(
            "\n".join([runner.TESTS[0] + "::test_one", *runner.E2E]) if name == "collection" else ""
        )
        err.write_text("")
        junit = next((arg.split("=", 1)[1] for arg in argv if arg.startswith("--junitxml=")), None)
        if junit:
            nodes = [arg for arg in argv if "::" in arg]
            Path(junit).write_text(
                "<testsuites><testsuite>"
                + "".join(
                    '<testcase classname="'
                    + node.split("::")[0][:-3].replace("/", ".")
                    + '" name="'
                    + node.split("::")[1]
                    + '"/>'
                    for node in nodes
                )
                + "</testsuite></testsuites>"
            )
        if name == "coverage":
            atomic_json(
                Path(work["raw"]) / "coverage.json",
                dict(files={p: dict(summary=dict(missing_lines=0)) for p in runner.OWNED}),
            )
        receipt = dict(
            name=name,
            passed=True,
            log_path=str(out),
            stdout_path=str(out),
            stderr_path=str(err),
            stdout_sha256=runner.reference(out)["sha256"],
            stderr_sha256=runner.reference(err)["sha256"],
        )
        atomic_json(out.with_suffix(".receipt.json"), receipt)
        return receipt

    monkeypatch.setattr(runner, "child", controlled_child)
    runner.validate(work, scratch)
    assert report.REQUIRED <= set(called)
    assert work["union_passed"] and work["coverage_complete"]
    value = report.build(work, tmp_path / (e.NAME + ".json"))
    assert value["count_controls_ready_score"] == 1 and value["verdict_class"] == "null"
    monkeypatch.setattr(runner, "validate", lambda w, s: None)
    monkeypatch.setattr(runner, "terminal", lambda p, r: dict(passed=True))
    assert runner.run(tmp_path / "production" / (e.NAME + ".json")) == 0
    monkeypatch.setattr(runner, "run", lambda *a, **k: 0)
    assert runner.main([]) == 0


def test_replay_receipts_and_byte_rejections(tmp_path):
    """REQ-VERIFY-8566: log identity and current code identity are immutable operands."""
    work = runner.prepare(e.ROOT, tmp_path / "raw")
    receipt = runner.child(
        "tiny", [sys.executable, "-c", "print('real child')"], tmp_path / "logs", deadline=15
    )
    work["receipts"] = [receipt]
    work_path = Path(work["raw"]) / "work.json"
    output = tmp_path / (e.NAME + ".json")

    def seal(changed):
        atomic_json(work_path, changed)
        atomic_json(output, report.build(changed, output))

    seal(work)
    assert report.replay(output)
    changed = deepcopy(work)
    changed["code"][0]["sha256"] = "wrong"
    seal(changed)
    assert not report.replay(output)
    changed = deepcopy(work)
    changed["receipts"][0]["duration_s"] += 1
    seal(changed)
    assert not report.replay(output)
    seal(work)
    Path(receipt["stdout_path"]).write_text("changed")
    assert not report.replay(output)
    Path(receipt["stdout_path"]).write_text("real child\n")
    original_events = json.loads(Path(work["evidence"]["events"]["path"]).read_bytes())
    original_events[0]["admitted"] = False
    atomic_json(Path(work["evidence"]["events"]["path"]), {"invalid": True})
    assert not report.replay(output)


def test_terminal_node_outcomes_reject_skip_and_loss(tmp_path):
    """REQ-VERIFY-8566: process success cannot hide a skipped or absent original node."""
    xml = tmp_path / "junit.xml"
    xml.write_text(
        '<testsuites><testsuite><testcase classname="tests.python.test_a" name="test_b"/></testsuite></testsuites>'
    )
    assert runner.terminal_nodes(xml) == [
        dict(node_id="tests/python/test_a.py::test_b", outcome="passed")
    ]
    xml.write_text(
        '<testsuites><testsuite><testcase classname="tests.python.test_a" name="test_b"><skipped/></testcase></testsuite></testsuites>'
    )
    assert runner.terminal_nodes(xml)[0]["outcome"] == "failed"


def test_rehashed_event_table_and_coverage_file_set(tmp_path):
    """REQ-VERIFY-8566: require every owned file and recompute rehashed event rows."""
    assert not runner.coverage_complete(
        dict(files={"a": dict(summary=dict(missing_lines=0))}), ["missing"]
    )
    work = runner.prepare(e.ROOT, tmp_path / "raw")
    events = Path(work["evidence"]["events"]["path"])
    rows = json.loads(events.read_bytes())
    rows[0]["admitted"] = False
    events.write_text(json.dumps(rows))
    work["evidence"]["events"] = runner.reference(events)
    atomic_json(Path(work["raw"]) / "work.json", work)
    output = tmp_path / (e.NAME + ".json")
    atomic_json(output, report.build(work, output))
    assert not report.replay(output)

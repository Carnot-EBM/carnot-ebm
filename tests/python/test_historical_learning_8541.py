"""REQ-REPORT-8541 and REQ-VERIFY-8541: original evidence cannot be replaced."""

import json
from pathlib import Path
import sys

import pytest

from carnot.reporting import historical_learning_8541 as e
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.v709_execution import child


def test_original_union_and_suite_authority(tmp_path):
    """SCENARIO-REPORT-8541-AUTHORITY: both objective suites need the same original seal."""
    old, nodes = e.original_union()
    assert old["experiment_id"] == 8530
    assert len(nodes) == len(set(nodes)) == 158
    authorities = e.authorities(nodes)
    reporting = "tests/python/test_objective_reporting_8461.py"
    assert authorities[reporting]["producer"]["path"].endswith(
        "8461_v728_objective_checkpoint_qualification.json"
    )
    assert authorities[reporting]["run_date"] == "20261012"
    with e.binding(reporting, tmp_path / "nested" / "scratch"):
        from carnot.reporting import objective_checkpoint_evidence_8461 as original

        actual = original.methods.authority(e.ROOT, tmp_path / "assessment")
        assert actual["activated"]
        assert any(t["id"] == original.TASK for t in actual["tasks"])
    with pytest.raises(ValueError, match="unknown_suite"):
        e.binding("foreign.py", tmp_path).__enter__()


def test_union_requires_actual_terminal_nodes(tmp_path):
    """SCENARIO-VERIFY-8541-UNION: outcomes, counts and exits qualify independently."""
    suite = "tests/python/test_objective_checkpoint_8461.py"
    node = suite + "::test_transition_controls_and_invalid_configuration"
    receipt = child(
        "real",
        [
            sys.executable,
            "-u",
            str(e.ROOT / e.CLI),
            "--suite",
            suite,
            *e.COMMON,
            node,
            "--basetemp=" + str(tmp_path / "pytest"),
        ],
        tmp_path,
        deadline=60,
    )
    receipt["collected_node_ids"] = [node]
    assert e.union([node], [receipt])["passed"]
    assert not e.union([node, node], [receipt])["passed"]
    receipt["timed_out"] = True
    assert not e.union([node], [receipt])["passed"]
    receipt["timed_out"] = False
    receipt["collected_node_ids"] = [node, node]
    assert not e.union([node], [receipt])["passed"]
    receipt["collected_node_ids"] = [node]
    Path(receipt["stdout_path"]).write_text(node + " FAILED [100%]\n")
    assert not e.union([node], [receipt])["passed"]
    assert not e.union([], [])["passed"]


def test_source_hash_and_cli_rejections(tmp_path):
    """SCENARIO-REPORT-8541-TERMINAL: missing and changed operands fail closed."""
    path = tmp_path / "bytes.json"
    atomic_json(path, {"real": True})
    ref = e.reference(path)
    assert e.read(ref) == {"real": True}
    path.write_text("{}")
    with pytest.raises(ValueError, match="source_hash"):
        e.read(ref)
    from carnot.reporting import historical_learning_runner_8541 as runner

    assert runner.main(["--deliberate-error"]) == 7
    assert runner.main(["--cold-replay", str(tmp_path / "absent")]) == 1
    with pytest.raises(SystemExit):
        runner.main(["--date", "20261014"])
    with pytest.raises(SystemExit):
        runner.main(["--private-control"])


def test_artifact_reconstruction_and_owned_failure(tmp_path):
    """SCENARIO-REPORT-8541-TERMINAL: honest failure replays; repaired summary claims do not."""
    from carnot.reporting import historical_learning_report_8541 as report

    node = "tests/python/private.py::test_real"
    receipt = child("failed", [sys.executable, "-c", "raise SystemExit(7)"], tmp_path / "logs")
    receipt["collected_node_ids"] = [node]
    work = dict(
        nodes=e.original_union()[1],
        receipts=[receipt],
        checks=[],
        failures=[],
        sources=[],
        authorities=e.authorities(e.original_union()[1]),
        preconditions={},
        start_ns=0,
        end_ns=1,
        historical_failure_reproduction=[],
        coverage=None,
    )
    atomic_json(tmp_path / "measurement.json", work)
    output = tmp_path / (e.NAME + ".json")
    value = report.build(work, tmp_path, output)
    assert value["verdict_class"] == "disqualified"
    assert value["historical_learning_validation_ready_score"] == 0
    assert value["censored_count"] == 158
    atomic_json(output, value)
    assert report.replay(output)
    value["historical_learning_validation_ready_score"] = 1
    value["reproducibility_checksum"] = report.checksum(value)
    atomic_json(output, value)
    assert not report.replay(output)
    assert not report.replay(tmp_path / "absent")


def test_real_worker_modes(tmp_path):
    """SCENARIO-REPORT-8541-AUTHORITY: script exits and family patches survive real dispatch."""
    suite = "tests/python/test_primary_publication_7928.py"
    source = tmp_path / "real.py"
    for text, expected in [
        ("pass", 0),
        ("raise SystemExit(7)", 7),
        ("raise SystemExit('error')", 1),
    ]:
        source.write_text(text)
        assert e.worker(suite, [str(source)], script=True) == expected
    source.write_text(
        "import subprocess,sys\nassert subprocess.run([sys.executable,'-c','pass']).returncode == 0\n"
    )
    assert e.worker(suite, [str(source)], script=True) == 0


def test_actual_private_cli_publication(tmp_path):
    """SCENARIO-REPORT-8541-TERMINAL: real private bytes, scanners and downstream readers run."""
    output = tmp_path / "results" / (e.NAME + ".json")
    receipt = child(
        "private_cli",
        [
            sys.executable,
            "-u",
            str(e.ROOT / e.CLI),
            "--date",
            "20261015",
            "--private-control",
            "--output",
            str(output),
        ],
        tmp_path / "cli",
        deadline=120,
    )
    assert receipt["passed"], Path(receipt["stderr_path"]).read_text()
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "disqualified"
    assert not value["historical_learning_validation_ready_score"]
    terminal = json.loads(Path(value["terminal_validation_sidecar_path"]).read_bytes())
    assert terminal["consumer"]["passed"]
    assert terminal["primary_sha256"] == e.reference(output)["sha256"]


def test_authority_pin_and_child_routing(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8541-AUTHORITY: corrupt snapshots fail before a real descendant starts."""
    original = e.sha256_file
    with monkeypatch.context() as scoped:
        scoped.setattr(
            e, "sha256_file", lambda p: "wrong" if p.name.startswith("design-") else original(p)
        )
        with pytest.raises(ValueError, match="original_authority_pin"):
            e.authorities(e.original_union()[1])
    source = tmp_path / "descendant.py"
    source.write_text(
        "import subprocess,sys\n"
        + f'assert subprocess.run([sys.executable,{str(e.ROOT / "scripts/experiments/experiment_8461_v728_objective_checkpoint_qualification.py")!r},"--deliberate-error"]).returncode == 7\n'
    )
    assert (
        e.worker("tests/python/test_objective_reporting_8461.py", [str(source)], script=True) == 0
    )


def test_deadline_collection_and_missing_original(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8541-UNION: an expired budget starts no child; missing inputs stay explicit."""
    from carnot.reporting import historical_learning_runner_8541 as runner

    work = dict(
        nodes=["suite::test"], authorities={"suite": {}}, start_ns=0, receipts=[], failures=[]
    )
    runner.families(work, [], tmp_path, tmp_path)
    assert not work["receipts"] and work["failures"][0]["check"] == "measurement_deadline"
    assert runner.collection_gate(["a"], ["b"], tmp_path)
    assert runner.collection_gate(["a"], ["a", "a"], tmp_path)
    assert not runner.collection_gate(["a"], ["a"], tmp_path)

    def absent():
        raise FileNotFoundError("original_missing")

    monkeypatch.setattr(e, "original_union", absent)
    output = tmp_path / "results" / (e.NAME + ".json")
    assert runner.run(output, tmp_path / "scratch") == 0
    assert json.loads(output.read_bytes())["gate_check_summary"][0]["field"] == "original_inputs"


def test_rehashed_log_receipt_and_authority_rejections(tmp_path):
    """SCENARIO-REPORT-8541-TERMINAL: source, log, receipt and authority claims have separate checks."""
    from copy import deepcopy
    from carnot.reporting import historical_learning_report_8541 as report

    receipt = child("evidence", [sys.executable, "-c", 'print("real")'], tmp_path / "logs")
    work = dict(
        nodes=[],
        receipts=[],
        checks=[receipt],
        failures=[],
        sources=[],
        authorities={},
        preconditions={},
        start_ns=0,
        end_ns=1,
        historical_failure_reproduction=[],
        coverage=None,
    )
    output = tmp_path / (e.NAME + ".json")
    for mode in ("source", "log", "receipt", "authority"):
        changed = deepcopy(work)
        if mode == "log":
            changed["checks"][0]["stdout_sha256"] = "wrong"
        elif mode == "receipt":
            changed["checks"][0]["duration_s"] += 1
        elif mode == "authority":
            changed["authorities"] = {"foreign": {}}
        atomic_json(tmp_path / "measurement.json", changed)
        value = report.build(changed, tmp_path, output)
        if mode == "source":
            value["code_config_hashes"][0]["sha256"] = "wrong"
        value["reproducibility_checksum"] = report.checksum(value)
        atomic_json(output, value)
        assert not report.replay(output)


def test_primitive_union_and_resource_tampering(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8541-TERMINAL: repaired primitive hashes cannot change the pinned union."""
    from carnot.reporting import historical_learning_report_8541 as report
    from carnot.reporting import historical_learning_runner_8541 as runner

    work = dict(
        nodes=["forged"],
        receipts=[],
        checks=[],
        failures=[],
        sources=[],
        authorities={},
        preconditions={},
        start_ns=0,
        end_ns=1,
        historical_failure_reproduction=[],
        coverage=None,
    )
    atomic_json(tmp_path / "measurement.json", work)
    output = tmp_path / (e.NAME + ".json")
    atomic_json(output, report.build(work, tmp_path, output))
    assert not report.replay(output)
    monkeypatch.setattr(runner, "filesystem_type", lambda path: "tmpfs")
    output = tmp_path / "blocked" / (e.NAME + ".json")
    assert runner.run(output, tmp_path / "scratch", control=True) == 0
    assert json.loads(output.read_bytes())["verdict_class"] == "blocked"


def test_incomplete_genuine_coverage_is_a_failed_check(tmp_path):
    """SCENARIO-VERIFY-8541-UNION: a real incomplete trace cannot pass the coverage gate."""
    from carnot.reporting import historical_learning_runner_8541 as runner

    raw = tmp_path / "coverage"
    raw.mkdir()
    config = raw / "coverage.ini"
    config.write_text(
        "[run]\ncore=sysmon\nparallel=true\ndata_file="
        + str(raw / ".coverage")
        + "\ninclude=\n    "
        + str(e.ROOT / e.OWNED[0])
        + "\n"
    )
    receipt = child(
        "trace",
        [
            sys.executable,
            "-m",
            "coverage",
            "run",
            "--rcfile=" + str(config),
            str(e.ROOT / e.CLI),
            "--deliberate-error",
        ],
        raw,
        expected=7,
    )
    assert receipt["passed"]
    work = dict(checks=[], coverage=None)
    runner.verification(work, raw, config, collect_current=False)
    assert work["coverage"]
    assert not next(r for r in work["checks"] if r["name"] == "coverage_report")["passed"]
    runner.verification(dict(checks=[], coverage=None), raw, config)


def test_genuine_print_children_cannot_impersonate_pytest(tmp_path):
    """SCENARIO-VERIFY-8541-UNION: real exit and logs alone cannot replace the canonical suite command."""
    node = e.original_union()[1][0]
    receipt = child("forged_family", [sys.executable, "-c", f'print({node!r}+" PASSED")'], tmp_path)
    receipt["collected_node_ids"] = [node]
    assert not e.union([node], [receipt])["passed"]


def test_required_checks_and_coverage_scope(tmp_path):
    """SCENARIO-REPORT-8541-TERMINAL: complete coverage and required commands have independent contracts."""
    from carnot.reporting import historical_learning_report_8541 as report

    assert not report.check_contract(dict(nodes=[], checks=[], preconditions={}), tmp_path)
    assert not report.complete_coverage(None)
    path = tmp_path / "unrelated.json"
    atomic_json(path, dict(totals=dict(num_statements=1, missing_lines=0), files={}))
    assert not report.complete_coverage(e.reference(path))


def test_partial_owned_coverage_and_relabelled_checks(tmp_path):
    """SCENARIO-REPORT-8541-TERMINAL: repaired scope totals and relabelled real receipts remain invalid."""
    from carnot.reporting import historical_learning_report_8541 as report
    from carnot.reporting import historical_learning_runner_8541 as runner

    raw = tmp_path / "partial"
    raw.mkdir()
    config = raw / "coverage.ini"
    config.write_text(
        "[run]\ncore=sysmon\nparallel=true\ndata_file="
        + str(raw / ".coverage")
        + "\ninclude=\n"
        + "".join("    " + str(e.ROOT / p) + "\n" for p in e.OWNED)
    )
    trace = child(
        "partial",
        [
            sys.executable,
            "-m",
            "coverage",
            "run",
            "--rcfile=" + str(config),
            str(e.ROOT / e.CLI),
            "--deliberate-error",
        ],
        raw,
        expected=7,
    )
    assert trace["passed"]
    assert child(
        "combine", [sys.executable, "-m", "coverage", "combine", "--rcfile=" + str(config)], raw
    )["passed"]
    path = raw / "coverage.json"
    assert child(
        "json",
        [sys.executable, "-m", "coverage", "json", "--rcfile=" + str(config), "-o", str(path)],
        raw,
    )["passed"]
    assert not report.complete_coverage(e.reference(path))
    data = json.loads(path.read_bytes())
    for row in data["files"].values():
        row["missing_lines"] = []
        row["summary"]["missing_lines"] = 0
    atomic_json(path, data)
    assert not report.complete_coverage(e.reference(path))
    work = dict(nodes=[], checks=[], preconditions={}, start_ns=0, end_ns=1)
    work["checks"] = [dict(trace, name=name) for name in runner.required_commands(work, raw)]
    assert not report.check_contract(work, raw)
    import coverage

    analyzer = coverage.Coverage(config_file=False, data_file=None)
    for label, row in data["files"].items():
        row["executed_lines"] = analyzer.analysis2(str(e.ROOT / label))[1]
        row["summary"]["num_statements"] = 0
    atomic_json(path, data)
    assert not report.complete_coverage(e.reference(path))
    expected = runner.required_commands(work, raw)
    work["checks"] = [
        dict(
            trace,
            name=name,
            argv=command,
            actual_exit=status,
            exit_code=status,
            expected_exit=status,
        )
        for name, (command, status) in expected.items()
    ]
    assert not report.check_contract(work, raw)

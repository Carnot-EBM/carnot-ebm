"""REQ-VERIFY-8615: actual script-path children cover private publication and rejection."""

import json
from pathlib import Path
import sys

from carnot.reporting import v740_contract_methods as e
from carnot.reporting import v740_contract_runner as runner
from carnot.reporting.v709_execution import child


def test_private_cli(tmp_path):
    """SCENARIO-VERIFY-8615-REPLAY: real publication and cold children retain terminal logs."""
    output = tmp_path / "results" / (e.NAME + ".json")
    common = [sys.executable, "-u", str(e.ROOT / e.CLI)]
    r = child(
        "private",
        common
        + [
            "--date",
            "20261017",
            "--root",
            str(tmp_path / "absent"),
            "--output",
            str(output),
            "--private-control",
        ],
        tmp_path / "logs",
        deadline=300,
    )
    assert r["passed"], Path(r["stderr_path"]).read_text()
    v = json.loads(output.read_bytes())
    assert v["verdict_class"] == "blocked"
    from carnot.reporting.primary_publication import read_bound_sidecar

    assert read_bound_sidecar(output, Path(v["terminal_validation_sidecar_path"]))[
        "primary_sha256"
    ] == e.sha256_file(output)
    assert child("cold", common + ["--cold-replay", str(output)], tmp_path / "logs", deadline=300)[
        "passed"
    ]
    v["completed_count"] += 1
    v["reproducibility_checksum"] = e.checksum(v)
    e.atomic_json(output, v)
    assert child(
        "tamper",
        common + ["--cold-replay", str(output)],
        tmp_path / "logs",
        deadline=300,
        expected=1,
    )["passed"]
    assert child("date", common + ["--date", "bad"], tmp_path / "logs", deadline=30, expected=2)[
        "passed"
    ]
    assert child(
        "owned", common + ["--deliberate-error"], tmp_path / "logs", deadline=30, expected=7
    )["passed"]
    assert child(
        "unsafe", common + ["--private-control"], tmp_path / "logs", deadline=30, expected=2
    )["passed"]


def test_validation_plan(tmp_path, monkeypatch):
    """REQ-VERIFY-8615: only scoped checks and exact original node unions are authorized."""
    monkeypatch.setattr(
        runner.collection,
        "collect",
        lambda private, suite: (
            [suite + "::private"],
            dict(name="collect_" + Path(suite).stem, argv=[]),
        ),
    )
    plan = runner.manifest(tmp_path)
    assert {"ruff_check", "ruff_format", "strict_mypy", "spec_coverage", "coverage_report"} <= {
        r["name"] for r in plan
    }
    assert all("tests/python" not in r["argv"] for r in plan)
    assert next(r for r in plan if r["name"] == "owned_tests")["deadline"] == 600


def test_validation_scratch_rejects_checkout():
    """SCENARIO-VERIFY-8615-REPLAY: child scratch must not collide with guarded result paths."""
    import pytest

    with pytest.raises(ValueError, match="private"):
        runner.manifest(e.ROOT / "results/raw/forbidden-validation-scratch")


def test_date_width():
    """REQ-VERIFY-8615: valid later dates pass without accepting short or invalid dates."""
    import argparse
    import pytest

    assert runner.date_value("20261018") == "20261018"
    for value in ["202611", "20260230"]:
        with pytest.raises(argparse.ArgumentTypeError):
            runner.date_value(value)


def test_exact_node_union(tmp_path):
    """REQ-VERIFY-8615: every collected original ID must execute exactly once."""
    xml = tmp_path / "nodes.xml"
    xml.write_text(
        '<testsuites><testsuite><testcase classname="tests.python.test_x" name="test_a"/></testsuite></testsuites>'
    )
    nodes = ["tests/python/test_x.py::test_a"]
    plan = [
        dict(name="collect_x", collected_node_ids=nodes),
        dict(name="owned_tests", collected_node_ids=nodes, junit_path=str(xml)),
    ]
    assert runner.exact_union(plan)["passed"]
    xml.write_text(
        '<testsuites><testsuite><testcase classname="tests.python.test_x" name="test_b"/></testsuite></testsuites>'
    )
    assert not runner.exact_union(plan)["passed"]
    xml.unlink()
    assert not runner.exact_union(plan)["passed"]


def test_owned_child_failure(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8615-REPLAY: an actual failed child remains disqualified."""
    output = tmp_path / (e.NAME + ".json")

    def failed_plan(private):
        e.atomic_json(private / "coverage.json", dict(fixture=True))
        e.atomic_json(private / "original_node_union.json", dict(suites={}))
        return [
            dict(
                name="owned_failure",
                argv=[sys.executable, "-c", "raise SystemExit(3)"],
                deadline=10,
                expected=0,
                scope="owned",
            )
        ]

    monkeypatch.setattr(runner, "manifest", failed_plan)
    monkeypatch.setattr(runner.shipped, "controls", lambda value, raw: [])
    monkeypatch.setattr(
        runner.shipped, "publish", lambda value, path, raw: e.atomic_json(path, value)
    )
    assert (
        runner.main(
            ["--date", "20261018", "--root", str(tmp_path / "absent"), "--output", str(output)]
        )
        == 0
    )
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "disqualified"
    assert not value["required_checks_passed"]

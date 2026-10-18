"""REQ-VERIFY-8646: exact unions and separate current statement coverage."""

import json
from pathlib import Path

import pytest

from carnot.reporting import installed_compatibility_8646 as e
from carnot.reporting import installed_validation_8646 as v
from carnot.reporting.current_work_receipt import atomic_json


def test_coverage_presence_and_missing_lines():
    """SCENARIO-VERIFY-8646-CUSTODY: absent current files cannot receive legacy credit."""
    assert not v.coverage_result({})["passed"]
    files = {p: dict(summary=dict(num_statements=1), missing_lines=[]) for p in e.OWNED}
    assert v.coverage_result(files)["passed"]
    files[e.OWNED[0]]["missing_lines"] = [1]
    assert not v.coverage_result(files)["passed"]


def test_validation_empty_nodes_and_failed_checks(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8646-CUSTODY: failed collection never invents JUnit outcomes."""

    def fake(name, argv, logs, **kwargs):
        logs.mkdir(parents=True, exist_ok=True)
        path = logs / (name + ".stdout")
        path.write_text("")
        return dict(name=name, passed=False, stdout_path=str(path))

    monkeypatch.setattr(v, "child", fake)
    work = dict(bundle={}, original_nodes=[], overlay=[], raw_shard_hashes=[])
    checks = v.validate(work, tmp_path)
    assert not next(r for r in checks if r["name"] == "exact_node_union")["passed"]
    assert not next(r for r in checks if r["name"] == "owned_coverage")["passed"]


def test_validation_complete_control_flow(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8646-CUSTODY: disjoint execution retains separate raw copies."""
    history = e.load(dict(path=str(e.ROOT / e.HISTORY), sha256=e.HISTORY_PIN))
    original = history["original_node_union"]["original_nodes"]
    current = e.TESTS[0] + "::test_new"
    nodes = original + [current, "tests/python/test_primary_publication_7928.py::test_reader"]
    outcomes = {}

    def fake(name, argv, logs, **kwargs):
        logs.mkdir(parents=True, exist_ok=True)
        path = logs / (name + ".stdout")
        path.write_text("\n".join(nodes) if name == "collect" else "")
        if name.startswith("nodes-"):
            selected = [arg for arg in argv if arg.startswith("tests/python/")]
            xml = Path(next(arg.split("=", 1)[1] for arg in argv if arg.startswith("--junitxml=")))
            xml.write_text("<testsuite/>")
            outcomes[str(xml)] = selected
            data = Path(
                next(arg.split("=", 1)[1] for arg in argv if arg.startswith("COVERAGE_FILE="))
            )
            data.with_name(".coverage.actual-child").write_bytes(
                b"orchestration fixture; not measurement"
            )
            (logs / ("imports-" + name.split("-")[1] + ".json")).write_text("{}")
        if name == "coverage_json":
            atomic_json(
                logs / "coverage.json",
                dict(
                    files={
                        p: dict(summary=dict(num_statements=1), missing_lines=[]) for p in e.OWNED
                    }
                ),
            )
        return dict(name=name, passed=True, stdout_path=str(path))

    monkeypatch.setattr(v, "child", fake)
    monkeypatch.setattr(v, "seal_authority", lambda path, **kwargs: dict(root=str(path)))
    monkeypatch.setattr(v.old, "original_overlay", lambda *args: ["original-overlay"])
    monkeypatch.setattr(v, "xml_nodes", lambda path: outcomes[str(path)])
    work = dict(original_nodes=original)
    checks = v.validate(work, tmp_path)
    assert all(row["passed"] for row in checks)
    assert work["original_node_union"]["passed"]
    assert work["current_node_union"]["passed"]
    assert {r["historical"] for r in work["child_shard_manifest"]} == {True, False}

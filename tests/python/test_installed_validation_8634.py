"""REQ-VERIFY-8634: validators retain exact nodes and original child coverage."""

import json
from pathlib import Path

from carnot.reporting import installed_compatibility_8634 as e
from carnot.reporting import installed_validation_8634 as v
from carnot.reporting.current_work_receipt import atomic_json


def test_missing_coverage_fails():
    """SCENARIO-VERIFY-8634-INSTALLED: omitted statements never qualify as zero misses."""
    assert not v.coverage_result({})["passed"]
    files = {
        p: dict(executed_lines=[1], missing_lines=[], summary=dict(num_statements=1))
        for p in v.COVERED
    }
    assert v.coverage_result(files)["passed"]
    files[v.COVERED[0]]["missing_lines"] = [2]
    assert not v.coverage_result(files)["passed"]


def test_validation_plan_and_missing_results(tmp_path, monkeypatch):
    """REQ-VERIFY-8634: disjoint children cannot invent missing JUnit or coverage."""
    seen = []
    node = e.TESTS[0] + "::test_private"

    def fake(name, argv, logs, **kwargs):
        logs.mkdir(parents=True, exist_ok=True)
        seen.append((name, argv))
        out = logs / (name + ".stdout")
        out.write_text(node + "\n" if name == "collect" else "")
        return dict(name=name, passed=True, stdout_path=str(out))

    monkeypatch.setattr(v, "child", fake)
    monkeypatch.setattr(v, "seal_authority", lambda path, **kwargs: dict(root=str(tmp_path)))
    receipts = v.validate(dict(bundle=dict(original_nodes=[node])), tmp_path)
    assert not next(r for r in receipts if r["name"] == "exact_node_union")["passed"]
    assert not next(r for r in receipts if r["name"] == "owned_coverage")["passed"]
    assert all(
        "--no-cov" in argv and "-n" in argv for name, argv in seen if name.startswith("nodes_")
    )
    assert any("--keep" in argv for _, argv in seen)


def test_validation_complete_reports(tmp_path, monkeypatch):
    """REQ-VERIFY-8634: combined reports retain original shards and every passed node."""
    node = "tests/python/test_resident_provenance_8593.py::test_private"
    node2 = "tests/python/test_lean_runner_8609.py::test_private"

    def fake(name, argv, logs, **kwargs):
        logs.mkdir(parents=True, exist_ok=True)
        out = logs / (name + ".stdout")
        out.write_text(node + "\n" + node2 + "\n" if name == "collect" else "")
        for arg in argv:
            if arg.startswith("--junitxml="):
                Path(arg.split("=", 1)[1]).write_text("<testsuite/>")
        if name == "coverage_json":
            atomic_json(
                tmp_path / "validation/coverage.json",
                dict(
                    files={
                        p: dict(
                            executed_lines=[1], missing_lines=[], summary=dict(num_statements=1)
                        )
                        for p in v.COVERED
                    }
                ),
            )
            (tmp_path / "validation/.coverage.child").write_bytes(b"unit fixture")
        return dict(name=name, passed=True, stdout_path=str(out))

    monkeypatch.setattr(v, "child", fake)
    monkeypatch.setattr(v, "xml_nodes", lambda path: [node] if "nodes_0" in path.name else [node2])
    monkeypatch.setattr(v, "seal_authority", lambda path, **kwargs: dict(root=str(tmp_path)))
    work = dict(bundle=dict(original_nodes=[node, node2]))
    receipts = v.validate(work, tmp_path)
    overlay = v.original_overlay(tmp_path / "overlay", [])
    assert "--ro-bind" in overlay and overlay[0] == "bwrap"
    assert "--dev-bind" in overlay and "--proc" in overlay
    assert (
        e.reference(e.ROOT / "python/carnot/inference/resident_protocol.py")["sha256"]
        != v.ORIGINAL_PROTOCOL_HASH
    )
    assert next(r for r in receipts if r["name"] == "owned_coverage")["passed"]
    assert work["child_shard_manifest"]
    assert not next(r for r in receipts if r["name"] == "exact_node_union")[
        "passed"
    ]  # 122 required
    atomic_json(
        tmp_path / "validation/collect.receipt.json",
        dict(name="collect", passed=True, stdout_path=str(tmp_path / "validation/collect.stdout")),
    )
    resumed = v.validate(work, tmp_path / "resumed", prior=tmp_path / "validation")
    assert resumed
    assert list((tmp_path / "resumed/validation").glob(".coverage.inherited-*"))


def test_worker_forced_cleanup(tmp_path, monkeypatch):
    """REQ-VERIFY-8634: a killed child fails validation even when its ready frame exists."""
    import pytest
    from carnot.reporting import installed_worker_8634 as w

    class Process:
        def __enter__(self):
            self.stdin = open(tmp_path / "input", "w+b")
            self.stdout = open(tmp_path / "output", "w+b")
            self.returncode = None
            return self

        def __exit__(self, *args):
            self.stdin.close()
            self.stdout.close()

        def poll(self):
            return self.returncode

        def kill(self):
            self.returncode = -9

        def wait(self, **kwargs):
            return self.returncode

    monkeypatch.setattr(w, "KINDS", ["expired"])
    monkeypatch.setattr(w.subprocess, "Popen", lambda *a, **kw: Process())
    replies = iter([b'{"status":"ready"}', b'{"status":"ok"}'])
    monkeypatch.setattr(w.wire, "read_frame", lambda *a: next(replies))
    monkeypatch.setattr(w.wire, "write_frame", lambda *a, **kw: None)
    monkeypatch.setattr(w.os, "write", lambda *a: (_ for _ in ()).throw(ValueError("failure")))
    with pytest.raises(ValueError, match="failure"):
        w.checks(["private"], dict(rows=[{}]), tmp_path, tmp_path, "python")

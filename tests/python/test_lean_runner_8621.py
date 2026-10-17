"""REQ-VERIFY-8621: bounded orchestration and rejected publication are explicit."""

from copy import deepcopy
import json
from pathlib import Path

from carnot.reporting import lean_qualification_8621 as e
from carnot.reporting import lean_runner_8621 as r
from carnot.reporting.current_work_receipt import atomic_json


def test_historical_design_and_activation_are_both_sealed(tmp_path):
    """REQ-VERIFY-8621: original assertions need original design as well as task bytes."""
    from carnot.reporting import lean_validation_8621 as v
    from carnot.reporting import v738_contract_methods as methods

    receipt = v.seal_authority(tmp_path / "original")
    assert methods.authority(Path(receipt["root"]))["ready"]
    assert receipt["design"]["sha256"] == receipt["replay_design"]["sha256"]
    assert receipt["activation"]["sha256"] == receipt["replay_activation"]["sha256"]
    assert receipt["activation"]["sha256"] == receipt["staged_activation"]["sha256"]
    for test_path in e.previous.TESTS:
        assert (Path(receipt["root"]) / test_path).read_bytes() == (e.ROOT / test_path).read_bytes()
    live = v.seal_authority(tmp_path / "live_source", live_source=True)
    assert live["replay_activation"]["sha256"] == live["live_activation"]["sha256"]
    assert live["activation"]["sha256"] != live["live_activation"]["sha256"]


def test_bounded_private_run(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8621-CURRENT: real private publication keeps the authority block."""
    work = e.previous.measure(tmp_path / "work", "20261017", root=tmp_path)
    work.update(
        invocation_task_id=e.TASK,
        current_methods_reference=e.METHODS_REFERENCE,
        current_gate_check_summary=[dict(check="missing", observed="missing")],
        historical_replay_rejection_rows=[],
    )
    atomic_json(Path(work["raw"]) / "work.json", work)
    monkeypatch.setattr(r.e, "measure", lambda *a: deepcopy(work))
    monkeypatch.setattr(r, "validate", lambda *a: [dict(name="unit", passed=True)])
    output = tmp_path / (e.NAME + ".json")
    assert r.main(["--date", "20261017", "--output", str(output)]) == 0
    assert r.main(["--cold-replay", str(output)]) == 0
    assert r.main(["--date", "bad"]) == 1
    assert r.main(["--date", "2026101"]) == 1
    clock = iter([0.0, 3600.0])
    monkeypatch.setattr(r, "monotonic", lambda: next(clock))
    assert r.main(["--date", "20261017", "--output", str(output)]) == 1


def test_collection_failure_never_expands_suite(tmp_path, monkeypatch):
    """REQ-VERIFY-8621: no collection is a failing empty union, never whole-repo pytest."""
    from carnot.reporting import lean_validation_8621 as v

    commands = []
    nodes = [
        "tests/python/test_resident_provenance_8593.py::test_fixture",
        e.TESTS[0] + "::test_fixture",
    ]

    def child(name, argv, logs, **kwargs):
        commands.append(argv)
        logs.mkdir(parents=True, exist_ok=True)
        out = logs / (name + ".stdout")
        out.write_text("\n".join(nodes) if name == "collection" else "")
        for arg in argv:
            if arg.startswith("--junitxml="):
                Path(arg.split("=", 1)[1]).write_text("<testsuite/>")
        if name == "coverage_json":
            atomic_json(Path(argv[-1]), dict(files={}))
        if name == "coverage_combine":
            rcfile = Path(next(a.split("=", 1)[1] for a in argv if a.startswith("--rcfile=")))
            (rcfile.parent / ".coverage").write_bytes(b"unit fixture only")
        return dict(name=name, passed=True, stdout_path=str(out))

    monkeypatch.setattr(v, "child", child)
    monkeypatch.setattr(v, "seal_authority", lambda root, **kw: dict(root=str(tmp_path)))
    monkeypatch.setattr(v, "xml_nodes", lambda p: [nodes[int(p.stem.split("-")[-1])]])
    work = {}
    receipts = v.validate(work, tmp_path / "raw")
    assert any(not row["passed"] for row in receipts)
    assert not any(c[-1] == "pytest" for c in commands)
    nodes.clear()
    assert any(not row["passed"] for row in v.validate({}, tmp_path / "empty"))

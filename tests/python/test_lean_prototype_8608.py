"""REQ-REPORT-8608: immutable inputs, cold replay and honest prototype readiness."""

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot.reporting import lean_resident_8608 as e
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    return e.measure(tmp_path_factory.mktemp("lean8608"), "20261017")


def test_executable_prototype(work, tmp_path):
    """SCENARIO-REPORT-8608-PROTOTYPE: private children prove readiness only."""
    value = e.artifact(
        work, [{"name": "fixture_check", "passed": True}], tmp_path / (e.NAME + ".json")
    )
    assert value["lean_prototype_ready_score"] == 1
    assert value["verdict_class"] == "null"
    assert value["independent_generalization_score"] == 0
    assert value["accepted_benchmark_claim"] is False
    path = tmp_path / "candidate.json"
    atomic_json(path, value)
    e.replay(path)
    value["completed_count"] += 1
    value["reproducibility_checksum"] = e.checksum(value)
    atomic_json(path, value)
    with pytest.raises(ValueError):
        e.replay(path)


def test_failed_owned_and_external_missing(work, tmp_path):
    """REQ-REPORT-8608: owned failures disqualify; missing authority blocks."""
    output = tmp_path / (e.NAME + ".json")
    failed = e.artifact(work, [{"name": "required", "passed": False}], output)
    assert failed["verdict_class"] == "disqualified"
    assert failed["lean_prototype_ready_score"] == 0
    missing = e.measure(tmp_path / "missing", "20261017", root=tmp_path)
    blocked = e.artifact(missing, [], output)
    assert blocked["verdict_class"] == "blocked"
    assert blocked["gate_check_summary"][0]["observed"] == "missing"
    with pytest.raises((OSError, ValueError)):
        e.replay(tmp_path / "absent")
    bad = deepcopy(failed)
    bad["lean_prototype_ready_score"] = 1
    bad["reproducibility_checksum"] = e.checksum(bad)
    path = tmp_path / "failed.json"
    atomic_json(path, bad)
    with pytest.raises(ValueError):
        e.replay(path)


def test_reference_and_semantic_tamper(work, tmp_path):
    """REQ-REPORT-8608: a repaired envelope cannot authorize different semantics."""
    path = tmp_path / "data.json"
    path.write_text("{}")
    ref = e.reference(path)
    e.authenticate(ref)
    path.write_text("[]")
    with pytest.raises(ValueError, match="input_hash"):
        e.authenticate(ref)
    value = e.artifact(work, [{"name": "required", "passed": True}], tmp_path / (e.NAME + ".json"))
    value["target_model"]["parameter_count"] = 108
    value["target_model"]["descriptor_sha256"] = canonical_hash(
        {k: v for k, v in value["target_model"].items() if k != "descriptor_sha256"}
    )
    value["reproducibility_checksum"] = e.checksum(value)
    atomic_json(path, value)
    with pytest.raises(ValueError):
        e.replay(path)


def test_runner_replay_and_failures(work, tmp_path, monkeypatch):
    """REQ-REPORT-8608: real replay CLI rejects missing and rehashed candidates."""
    import subprocess
    import sys
    from carnot.reporting import lean_runner_8608 as runner

    output = tmp_path / (e.NAME + ".json")
    value = e.artifact(work, [{"name": "required", "passed": True}], output)
    atomic_json(output, value)
    for args, expected in [
        (["--cold-replay", str(output)], 0),
        (["--cold-replay", str(tmp_path / "missing")], 1),
        (["--date", "bad"], 1),
    ]:
        result = subprocess.run(
            [sys.executable, "-u", str(e.ROOT / e.CLI), *args],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == expected, result.stderr
    assert runner.main(["--cold-replay", str(output)]) == 0
    assert runner.main(["--cold-replay", str(tmp_path / "missing")]) == 1
    assert runner.main(["--date", "bad"]) == 1
    monkeypatch.setattr(runner.e, "measure", lambda *a: work)
    monkeypatch.setattr(
        runner, "validate", lambda *a: [{"name": "unit_orchestration", "passed": True}]
    )
    assert runner.main(["--date", "20261017", "--output", str(output)]) == 0
    assert runner.main(["--date", "20261017", "--output", str(output)]) == 0


def test_validation_orchestration(tmp_path, monkeypatch):
    """REQ-REPORT-8608: commands retain original nodes and never expand an empty shard."""
    from carnot.reporting import lean_runner_8608 as runner

    nodes = [e.TESTS[0] + "::test_one", e.TESTS[1] + "::test_two", *runner.E2E]
    commands = []

    def child(name, argv, raw, **kwargs):
        commands.append(argv)
        raw.mkdir(parents=True, exist_ok=True)
        log = raw / (name + ".stdout")
        log.write_text("\n".join(nodes) if name == "collection" else "")
        for arg in argv:
            if arg.startswith("--junitxml="):
                Path(arg.split("=", 1)[1]).write_text("<testsuite/>")
        if name == "coverage_json":
            Path(argv[-1]).write_text(json.dumps(dict(totals=dict(percent_covered=100))))
        return dict(name=name, passed=True, stdout_path=str(log))

    monkeypatch.setattr(runner, "child", child)
    monkeypatch.setattr(runner, "xml_nodes", lambda p: [nodes[int(p.stem.split("-")[-1])]])
    receipts = runner.validate(tmp_path / "validation")
    assert all(r["passed"] for r in receipts)
    assert any("--fail-under=100" in c for c in commands)
    nodes.clear()
    receipts = runner.validate(tmp_path / "empty")
    assert any(not r["passed"] for r in receipts)
    assert runner.terminal(tmp_path / "missing", tmp_path)["passed"] is not False


@pytest.mark.parametrize(
    "kind", ["checksum", "baseline", "head", "response", "graph", "descriptor", "config", "trace"]
)
def test_rehashed_primitive_rejections(work, tmp_path, kind):
    """REQ-REPORT-8608: each primitive remains constrained by original operands."""
    changed = deepcopy(work)
    if kind == "baseline":
        path = tmp_path / "baseline.py"
        path.write_text("changed baseline")
        changed["baseline_snapshots"][0]["preserved"] = e.reference(path)
    elif kind == "head":
        changed["head"]["parameters"][0] += 1
    elif kind == "response":
        changed["sessions"][0]["responses"][0]["response"]["probability"] += 0.01
    elif kind in {"graph", "trace"}:
        row = changed["sessions"][-1]
        row["modules"].append("sklearn")
        if kind == "graph":
            path = tmp_path / "modules.json"
            atomic_json(path, row["modules"])
            row["graph_reference"] = e.reference(path)
    elif kind == "descriptor":
        changed["target_model"]["parameter_count"] = 108
    elif kind == "config":
        config = json.loads(Path(changed["config_reference"]["path"]).read_bytes())
        config["head"]["temperature"] = 99
        path = tmp_path / "config.json"
        atomic_json(path, config)
        changed["config_reference"] = e.reference(path)
    primitive = tmp_path / "primitive.json"
    atomic_json(primitive, changed)
    value = e.artifact(work, [{"name": "required", "passed": True}], tmp_path / (e.NAME + ".json"))
    value["work_reference"] = e.reference(primitive)
    if kind != "checksum":
        value["reproducibility_checksum"] = e.checksum(value)
    path = tmp_path / "candidate.json"
    atomic_json(path, value)
    with pytest.raises(ValueError):
        e.replay(path)


@pytest.mark.parametrize("kind", ["qualification", "publication", "head"])
def test_upstream_rejections(tmp_path, monkeypatch, kind):
    """REQ-REPORT-8608: qualification and original descriptor are separate gates."""
    for number, suffix in [
        (8593, "resident_provenance_qualification"),
        (8596, "resident_complete_cost"),
    ]:
        name = f"experiment_{number}_v738_{suffix}"
        value = json.loads((e.ROOT / "results" / (name + ".json")).read_bytes())
        if kind == "qualification":
            value["required_checks_passed"] = False
        if kind == "head" and number == 8593:
            value["target_model"]["parameter_count"] = 108
        path = tmp_path / "results" / (name + ".json")
        atomic_json(path, value)
        sidecar = (
            path.parent
            / "raw"
            / name
            / "validators"
            / (e.reference(path)["sha256"].split(":")[1] + ".json")
        )
        atomic_json(
            sidecar,
            dict(
                primary_sha256=e.reference(path)["sha256"],
                report=dict(passed=kind != "publication"),
            ),
        )
    if kind == "head":
        with pytest.raises(ValueError, match="head_descriptor"):
            e.measure(tmp_path / "raw", "20261017", root=tmp_path)
    else:
        assert (
            len(e.measure(tmp_path / "raw", "20261017", root=tmp_path)["gate_check_summary"]) == 2
        )


def test_session_rejects_failed_child(tmp_path, monkeypatch):
    """REQ-REPORT-8608: child failures and forced cleanup cannot become valid receipts."""
    import resident_protocol as wire

    config = tmp_path / "config.json"
    atomic_json(config, dict(rows=[]))
    with pytest.raises(ValueError, match="worker_ready"):
        e.session(config, "invalid", tmp_path / "failed")

    def expire(*args):
        raise TimeoutError("deliberate child timeout")

    monkeypatch.setattr(wire, "read_frame", expire)
    with pytest.raises(TimeoutError):
        e.session(config, "python", tmp_path / "timeout")

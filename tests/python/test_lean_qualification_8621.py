"""REQ-REPORT-8621 / REQ-VERIFY-8621: current authority cannot rewrite old evidence."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import lean_qualification_8621 as e
from carnot.reporting.current_work_receipt import atomic_json


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    return e.measure(tmp_path_factory.mktemp("v740-fresh"), "20261017")


def test_current_authority_and_preserved_history(work):
    """SCENARIO-REPORT-8621-CURRENT: fresh invocation binds matching immutable methods."""
    assert not work["current_gate_check_summary"]
    assert work["historical_replay_rejection_rows"][0]["passed"]
    assert work["current_binding"]["operands_equal"]
    assert all(r["passed"] for r in work["parity_rows"] + work["lifecycle_rows"])
    assert len([r for r in work["parity_rows"] if r["kind"] == "original64"]) == 256
    history, _ = e.previous.upstream(e.ROOT, e.previous.NAME)
    assert history["lean_measurement_ready_score"] == 0
    assert history["verdict_class"] == "blocked"


@pytest.mark.parametrize("kind", ["task", "methods", "producer", "head", "descriptor", "old"])
def test_binding_rejects_substitution(work, kind):
    """REQ-REPORT-8621: producer identity and every numerical operand remain sealed."""
    changed = deepcopy(work)
    ref, producers = deepcopy(e.METHODS_REFERENCE), deepcopy(e.producer_map())
    if kind == "task":
        changed["invocation_task_id"] = e.previous.TASK
    elif kind == "methods":
        ref["sha256"] = "sha256:wrong"
    elif kind == "producer":
        producers[e.TASK] = "results/wrong.json"
    elif kind == "head":
        changed["registered_lean_methods"]["requests"] = 63
    elif kind == "descriptor":
        changed["target_model"]["temperature"] = 99
    else:
        old, _ = e.previous.upstream(e.ROOT, e.previous.NAME)
        changed = json.loads(e.previous.authenticate(old["work_reference"]).read_bytes())
    with pytest.raises(ValueError):
        e.bind_current(changed, ref, producers)


def test_verdict_replay_and_controls(work, tmp_path):
    """SCENARIO-VERIFY-8621-CUSTODY: cold arithmetic and owned failure control readiness."""
    receipts = [dict(name=n, passed=True) for n in e.REQUIRED]
    output = tmp_path / (e.NAME + ".json")
    value = e.artifact(work, receipts, output)
    assert value["lean_measurement_ready_score"] == 0
    assert value["verdict_class"] == "disqualified"
    atomic_json(output, value)
    e.replay(output)
    assert e.artifact(work, [], output)["lean_measurement_ready_score"] == 0
    assert e.artifact(work, [dict(passed=False)], output)["verdict_class"] == "disqualified"
    blocked = deepcopy(work)
    blocked["current_gate_check_summary"] = [dict(check="missing", observed="missing")]
    assert e.artifact(blocked, receipts, output)["verdict_class"] == "blocked"
    for key in ["completed_count", "current_methods_reference", "lean_measurement_ready_score"]:
        changed = deepcopy(value)
        changed[key] = "tampered"
        changed["reproducibility_checksum"] = e.previous.checksum(changed)
        atomic_json(output, changed)
        with pytest.raises(ValueError):
            e.replay(output)
    changed["reproducibility_checksum"] = "wrong"
    atomic_json(output, changed)
    with pytest.raises(ValueError, match="checksum"):
        e.replay(output)
    with pytest.raises(OSError):
        e.replay(tmp_path / "missing")


def test_private_cli(work, tmp_path):
    """REQ-VERIFY-8621: direct CLI cold replay works outside checkout without PYTHONPATH."""
    output = tmp_path / (e.NAME + ".json")
    atomic_json(output, e.artifact(work, [], output))
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    for args, expected in [
        (["--cold-replay", str(output)], 0),
        (["--date", "bad"], 1),
        (["--date", "2026101"], 1),
        (["--cold-replay", str(tmp_path / "absent")], 1),
    ]:
        child = subprocess.run(
            [sys.executable, "-u", str(e.ROOT / e.CLI), *args],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            timeout=60,
        )
        assert child.returncode == expected, child.stderr


def test_missing_current_methods(work, tmp_path, monkeypatch):
    """REQ-REPORT-8621: a missing current authority is an external block, not zero."""
    monkeypatch.setattr(e.previous, "measure", lambda *a, **kw: deepcopy(work))
    missing = dict(path=str(tmp_path / "missing"), sha256=e.METHODS_REFERENCE["sha256"])
    changed = e.measure(tmp_path / "blocked", "20261017", methods_reference=missing)
    gate = changed["current_gate_check_summary"][0]
    assert gate["observed"] == "missing"
    assert gate["field"] == "methods_ready_score"


@pytest.mark.parametrize("kind", ["config", "binding", "union", "coverage"])
def test_rehashed_primitive_binding(work, tmp_path, kind, monkeypatch):
    """REQ-VERIFY-8621: replacing the primitive and its hash cannot replace authority."""
    changed = deepcopy(work)
    if kind == "config":
        config = json.loads(e.previous.authenticate(changed["config_reference"]).read_bytes())
        config["head"]["temperature"] = 99
        atomic_json(tmp_path / "config.json", config)
        changed["config_reference"] = e.previous.reference(tmp_path / "config.json")
        with pytest.raises(ValueError, match="operands"):
            e.bind_current(changed, e.METHODS_REFERENCE, e.producer_map())
        return
    if kind == "binding":
        changed["current_binding"]["operands_equal"] = False
    elif kind == "coverage":
        changed["child_statement_coverage"] = dict(passed=False)
        monkeypatch.setattr(e, "coverage_matches", lambda proof: False)
    else:
        changed["original_122_node_union"] = dict(collected=[], passed=True)
    atomic_json(tmp_path / "primitive.json", changed)
    value = e.artifact(work, [], tmp_path / (e.NAME + ".json"))
    value["work_reference"] = e.previous.reference(tmp_path / "primitive.json")
    value["reproducibility_checksum"] = e.previous.checksum(value)
    atomic_json(tmp_path / "candidate.json", value)
    with pytest.raises(ValueError):
        e.replay(tmp_path / "candidate.json")


def test_coverage_reduces_raw_statements(tmp_path):
    """REQ-VERIFY-8621: actual raw coverage defeats rehashed executable-line invention."""
    import coverage

    source = tmp_path / "source.py"
    source.write_text("result = (\n    1\n)\n")
    data = tmp_path / ".coverage"
    cov = coverage.Coverage(data_file=str(data), config_file=False)
    cov.start()
    exec(compile(source.read_text(), str(source), "exec"), {})
    cov.stop()
    cov.save()
    report = tmp_path / "coverage.json"
    cov.json_report(outfile=str(report))
    measured = json.loads(report.read_bytes())
    proof = dict(
        passed=False,
        data_references=[e.previous.reference(data)],
        report=e.previous.reference(report),
        files=measured["files"],
    )
    assert e.coverage_matches(proof)
    next(iter(measured["files"].values()))["missing_lines"] = [2]
    atomic_json(report, measured)
    proof["report"] = e.previous.reference(report)
    assert not e.coverage_matches(proof)

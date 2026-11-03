"""REQ-REPORT-8756, REQ-VERIFY-8756: scoped qualification preserves old failures."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from carnot.reporting import feature_reuse_8756 as q
from carnot.reporting.v709_execution import child


@pytest.fixture
def work(tmp_path):
    """A private reduction fixture allows each acceptance gate to fail independently."""
    raw = tmp_path / "raw"
    raw.mkdir()
    q.e.atomic_json(raw / "work.json", {})
    return dict(
        raw=str(raw),
        invocation_id="private-fixture",
        inputs=dict(gate_check_summary=[], references=[], original_nodes=[]),
        measurement=dict(
            rows=[dict(r, status="completed", passed=True) for r in q.e.roster()],
            passed=True,
        ),
        validation=dict(
            checks=[dict(name="scoped", passed=True)],
            receipts=[],
            coverage=dict(complete=True),
            children=[],
            repository_health_reference=dict(full_suite_pass=False),
            owned_failure_control=dict(passed=True, feature_reuse_ready_score=0),
        ),
        sources=[],
        preconditions={},
        owned_error=None,
        duration_s=0.01,
        started_at_utc="2026-11-03T00:00:00+00:00",
        ended_at_utc="2026-11-03T00:00:00.010000+00:00",
        phase_spans=[],
    )


def test_independent_health_and_failed_owned_gate(work, tmp_path):
    """SCENARIO-VERIFY-8756-FAILURE: only complete scoped checks can grant readiness."""
    value = q.build(work, tmp_path / (q.e.NAME + ".json"))
    assert value["experiment_id"] == 8756 and value["run_date"] == "20261103"
    assert value["feature_reuse_ready_score"] == 1
    assert value["verdict_class"] == "circular_positive"
    assert value["repository_health_reference"]["full_suite_pass"] is False
    assert value["scoped_validation_scope"]["repository_health_is_dependency"] is False
    assert value["numerical_training_calls"] == 0 and value["MODEL_SPECS"] == []
    assert value["cost_protocol_reference"]["sha256"] == q.PROTOCOL["sha256"]
    work["validation"]["checks"].append(dict(name="owned", passed=False))
    rejected = q.build(work, tmp_path / "unused")
    assert rejected["feature_reuse_ready_score"] == 0
    assert rejected["verdict_class"] == "disqualified"
    work["validation"]["checks"].pop()
    work["validation"]["coverage"]["complete"] = False
    assert q.build(work, tmp_path / "unused")["feature_reuse_ready_score"] == 0


def test_protocol_inputs_and_original_union(monkeypatch, tmp_path):
    """SCENARIO-REPORT-8756-QUALIFICATION: the exact122 and original timing design are sealed."""
    inputs = q.inputs(q.e.ROOT)
    assert inputs["passed"], inputs["gate_check_summary"]
    protocol = q.e.load(q.PROTOCOL)
    assert inputs["original_nodes"] == protocol["original_node_union"]
    assert len(inputs["original_nodes"]) == 122
    cost = protocol["original_cost_protocol"]
    assert (cost["process_cells"], cost["item_evaluations_per_cell"]) == (40, 1024)
    assert q.e.load(protocol["preserved_primary"])["feature_reuse_ready_score"] == 0
    assert not q.inputs(tmp_path)["passed"]
    original = q.BASE_INPUTS
    monkeypatch.setattr(q, "BASE_INPUTS", lambda root: dict(original(root), original_nodes=[]))
    with pytest.raises(ValueError, match="original122_protocol"):
        q.inputs(q.e.ROOT)
    sealed = {r["path"] for r in q.source_hashes()}
    assert all(str(q.e.ROOT / n.split("::")[0]) in sealed for n in inputs["original_nodes"])


def test_real_owned_failure_control(tmp_path, monkeypatch, work):
    """SCENARIO-VERIFY-8756-FAILURE: a real pytest assertion exits1 and zeros readiness."""
    raw, scratch = tmp_path / "checks", tmp_path / "scratch"
    raw.mkdir()
    scratch.mkdir()
    base = dict(
        coverage=dict(complete=True),
        checks=[dict(name="scoped", passed=True), dict(name="full_python_suite", passed=False)],
        receipts=[dict(name="full_python_suite", passed=False, stdout_path="health")],
        original_node_union=dict(passed=True),
    )
    monkeypatch.setattr(q.legacy, "validate", lambda *args: deepcopy(base))
    result = q.validate(raw, scratch, [], [])
    control = result["owned_failure_control"]
    assert control["passed"] and control["receipt"]["actual_exit"] == 1
    assert control["feature_reuse_ready_score"] == 0
    assert "AssertionError" in Path(control["receipt"]["stdout_path"]).read_text()
    assert all(r["passed"] for r in result["checks"])
    assert result["repository_health_reference"]["full_suite_pass"] is False
    work["validation"]["checks"].append(dict(name="real_owned_node", passed=False))
    assert q.build(work, tmp_path / "unused")["feature_reuse_ready_score"] == 0
    monkeypatch.setattr(q, "child", lambda *args, **kw: dict(actual_exit=0, passed=False))
    other = tmp_path / "other"
    other.mkdir()
    rejected = q.validate(other, scratch, [], [])
    assert not rejected["owned_failure_control"]["passed"]
    assert not all(r["passed"] for r in rejected["checks"])


def test_cli_private_publication_replay_and_rejection(tmp_path):
    """SCENARIO-VERIFY-8756-CUSTODY: real CLI children publish only sealed terminal bytes."""
    assert q.main(["--date", "wrong"]) == 2
    assert q.main(["--private-output", str(q.e.ROOT / "unsafe.json")]) == 2
    assert q.main(["--cold-replay", str(tmp_path / "missing")]) == 1
    output = tmp_path / (q.e.NAME + ".json")
    receipt = child(
        "blocked",
        [sys.executable, "-u", q.e.CLI, "--root", str(tmp_path), "--private-output", str(output)],
        tmp_path / "logs",
        deadline=120,
    )
    assert receipt["passed"], Path(receipt["stderr_path"]).read_text()
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked" and value["censored_count"] == 12
    assert value["feature_reuse_ready_score"] == 0
    authority = value["primitive_reference"]
    assert q.replay(output, authority) and not q.replay(output)
    assert (
        q.main(
            ["--cold-replay", str(output), "--authority", authority["path"], authority["sha256"]]
        )
        == 0
    )
    value["completed_count"] += 1
    q.e.atomic_json(output, value)
    assert not q.replay(output, authority)


def test_main_dispatch_keeps_historical_namespaces(monkeypatch):
    """REQ-VERIFY-8756: a new producer cannot rebind historical reports or validators."""
    from carnot.reporting import single_extraction_8743 as old

    before = old.NAME, old.TASK, old.OWNED[:], old.TESTS[:]
    monkeypatch.setattr(q.legacy.runtime, "run", lambda *args, **kw: 0)
    assert q.main([]) == 0
    assert before == (old.NAME, old.TASK, old.OWNED, old.TESTS)


def test_freeze_nodes_before_measurement(tmp_path, monkeypatch):
    """REQ-VERIFY-8756: missing original nodes prevent even the installed build."""
    raw, scratch = tmp_path / "raw", tmp_path / "scratch"
    raw.mkdir()
    scratch.mkdir()
    stdout = raw / "collection.stdout"
    node = "tests/python/original.py::test_original"
    stdout.write_text(node + "\n")
    receipt = dict(passed=True, stdout_path=str(stdout))
    monkeypatch.setattr(q, "child", lambda *args, **kw: receipt)

    def measured(folder, private, inputs, **kwargs):
        frozen = json.loads((raw / "frozen_test_nodes.json").read_text())
        assert frozen["node_ids"] == [node]
        assert frozen["input_sha256"] == q.e.canonical_hash(inputs)
        return dict(passed=True, traces=[], receipts=[])

    monkeypatch.setattr(q, "BASE_MEASURE", measured)
    result = q.measure(raw, scratch, dict(original_nodes=[node]))
    assert result["passed"] and result["frozen_test_node_reference"] in result["traces"]
    stdout.write_text("")
    with pytest.raises(ValueError, match="premeasurement_node_union"):
        q.measure(raw, scratch, dict(original_nodes=[node]))

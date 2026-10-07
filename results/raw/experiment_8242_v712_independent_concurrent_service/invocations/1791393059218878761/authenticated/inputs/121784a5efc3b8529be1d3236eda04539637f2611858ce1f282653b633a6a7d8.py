"""REQ-REPORT-8236 / REQ-VERIFY-8236: current evidence uses private fixtures."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import qualified_concurrency_8236 as q
from carnot.reporting import concurrency_execution_8227 as old
from carnot.verify import concurrency_canary_8227 as e


def test_actual_error_reducer(tmp_path):
    """SCENARIO-REPORT-8236-ERROR: reduce a real HTTP503 and durable cancellation."""
    observed = e.qualify(tmp_path / "http")
    assert observed["passed"] and observed["rows"][2]["status"] == "error"
    rows = observed["rows"]
    value = old.build(
        dict(protocol=dict(canary=rows), checks=[]), dict(rows=rows), tmp_path, [], 1, True
    )
    assert value["failed_count"] == 1 and value["completed_count"] == 3
    assert value["rows"][2]["output_token_count"] is None
    canceled = e.acquire(
        rows, lambda *a: pytest.fail("canceled work invoked"), tmp_path / "cancel", 2, cap_s=0
    )
    assert all(r["status"] == "censored" for r in canceled)
    assert not e.recorder.Journal(tmp_path / "cancel/events.jsonl").pending


def protocol():
    """SCENARIO-VERIFY-8236-BINDINGS: use authentic original obligations as inputs."""
    return json.loads((e.ROOT / old.PROTOCOL).read_text())


def test_binding_and_substrate(tmp_path):
    """REQ-REPORT-8236: immutable envelopes and planned models earn no live credit."""
    original = protocol()
    binding = q.bind(original, [])
    assert binding["config"] == original["config"]
    assert [r["envelope"] for r in binding["canary"]] == [r["envelope"] for r in original["canary"]]
    assert len({r["request_id"] for r in binding["canary"]}) == 8
    assert len({r["response_id"] for r in binding["canary"]}) == 8
    for bad in [
        dict(original, canary=[]),
        dict(original, benchmark=original["canary"]),
        dict(original, config={}),
    ]:
        with pytest.raises(ValueError):
            q.bind(bad, [])
    data = dict(protocol=binding, checks=[], coverage_statement_counts={})
    value = q.build(data, {}, tmp_path, [dict(passed=True)], 1, False)
    assert value["experiment_id"] == 8236 and value["inference_substrate_class"] == "no_model_load"
    assert value["concurrent_canary_ready_score"] == 0 and value["censored_count"] == 8
    assert value["model_invocation_counts"]["generation_calls"] == 0
    assert value["generalized_learning_benefit_score"] == 0


def cli(*args):
    """SCENARIO-REPORT-8236-CLI: children resolve this checkout from private cwd."""
    return subprocess.run(
        [sys.executable, str(e.ROOT / q.CLI), *map(str, args)],
        cwd="/tmp",
        env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
        capture_output=True,
        text=True,
        timeout=90,
    )


@pytest.mark.parametrize("fixture", [False, True])
def test_cli_and_replay(tmp_path, fixture):
    """SCENARIO-REPORT-8236-CLI: terminal rows survive negative cold replay controls."""
    output = tmp_path / (q.NAME + ".json")
    args = (
        ["--fixture-e2e", output]
        if fixture
        else ["--root", tmp_path / "missing", "--output", output]
    )
    child = cli(*args)
    assert child.returncode == 0, child.stdout + child.stderr
    value = json.loads(output.read_text())
    assert value["required_checks_passed"] and value["intended_count"] == 8
    assert value["verdict_class"] == ("circular_positive" if fixture else "blocked")
    assert value["concurrent_canary_ready_score"] == 0
    assert cli("--cold-replay", output).returncode == 0
    changed = deepcopy(value)
    changed["completed_count"] += 1
    changed["reproducibility_checksum"] = old.checksum(changed)
    e.atomic_json(output, changed)
    assert cli("--cold-replay", output).returncode == 1
    e.atomic_json(output, value)
    Path(value["replay_inputs"]["result_path"]).write_text("{} modified")
    assert cli("--cold-replay", output).returncode == 1
    assert cli("--date", "20261006").returncode == 2
    assert cli("--fixture-e2e", e.ROOT / "results/bad.json").returncode == 2


@pytest.mark.parametrize("mode", ["valid", "resource", "owned", "isolation", "terminal"])
def test_natural_orchestration(tmp_path, monkeypatch, mode):
    """REQ-VERIFY-8236: resource acquisition follows complete owned validation."""
    from test_concurrency_canary_8227 import fake_live
    from carnot.reporting import qualified_concurrency_execution_8236 as run
    from carnot.inference import concurrency_runtime_8227 as live

    p, resources = fake_live(tmp_path, monkeypatch)
    allocation = live.GpuLease.acquire
    owners = []

    def acquire(**kwargs):
        owners.append(kwargs["task_id"])
        return allocation(**kwargs)

    monkeypatch.setattr(live.GpuLease, "acquire", acquire)
    p = q.bind(p, [])
    natural = live.live(p, resources, tmp_path / "canary", task_id=q.TASK)
    assert owners == [q.TASK]
    natural["qualification"] = dict(passed=True)
    data = dict(ready=True, protocol=p, identity=p["identity"], checks=[], refs=[], code=[])
    monkeypatch.setattr(q, "inputs", lambda *a: deepcopy(data))
    sequence = []

    def plan(private):
        e.atomic_json(
            private / "coverage.json", dict(totals=dict(num_statements=1, covered_lines=1))
        )
        return [old.CommandSpec("owned", ("true",), "owned", 5)]

    def execute(commands, raw):
        sequence.extend(c.name for c in commands)
        return [
            dict(
                name=c.name,
                passed=not (
                    mode == "owned"
                    and c.scope == "owned"
                    or mode == "terminal"
                    and c.scope == "terminal"
                ),
                normal_exit=True,
            )
            for c in commands
        ]

    def preflight(*args):
        assert "owned" in sequence
        sequence.append("resources")
        return dict(
            resources,
            receipts=[],
            checks=[dict(passed=mode != "resource", artifact_field="private_gpu", observed=False)],
        )

    monkeypatch.setattr(q, "validation_plan", plan)
    monkeypatch.setattr(run, "execute", execute)
    monkeypatch.setattr(live, "preflight", preflight)
    monkeypatch.setattr(live, "command", lambda *a: ["fixture", "--port", "100"])
    monkeypatch.setattr(live, "live", lambda *a, **kw: natural)
    monkeypatch.setattr(e, "qualify", lambda *a: dict(passed=mode != "isolation"))
    monkeypatch.setattr(q, "BINDINGS", str(tmp_path / "bindings.json"))
    output = tmp_path / (q.NAME + ".json")
    e.atomic_json(output, dict(experiment_id=8236, task_id=q.TASK))
    assert run.main(["--output", str(output)]) == (1 if mode == "terminal" else 0)
    if mode != "terminal":
        value = json.loads(output.read_text())
        assert value["verdict_class"] == (
            "blocked"
            if mode == "resource"
            else "disqualified"
            if mode in {"owned", "isolation"}
            else "null"
        )
        assert value["coverage_statement_counts"]["covered_lines"] == 1
        if mode == "valid":
            for row in natural["rows"]:
                row["journal_path"] = str(
                    tmp_path / "canary" / ("canary_" + row["arm"]) / "requests/events.jsonl"
                )
            raw = Path(value["replay_inputs"]["data_path"]).parent
            e.atomic_json(raw / "result.json", natural)
            value["raw_shard_hashes"] = []
            value["reproducibility_checksum"] = old.checksum(value)
            e.atomic_json(output, value)
            assert q.replay(output)
            natural["rows"][0]["clocks"]["issue"] += 1
            e.atomic_json(raw / "result.json", natural)
            assert not q.replay(output)
    if mode in {"owned", "isolation"}:
        assert "resources" not in sequence


def test_replay_checksum_and_missing(tmp_path):
    """SCENARIO-REPORT-8236-CLI: missing primitives and changed checksum fail closed."""
    path = tmp_path / (q.NAME + ".json")
    assert not q.replay(path)
    value = q.build(dict(checks=[]), {}, tmp_path, [], 1, False)
    value["reproducibility_checksum"] = "changed"
    e.atomic_json(path, value)
    assert not q.replay(path)


def test_protocol_envelope_drift():
    """SCENARIO-VERIFY-8236-BINDINGS: unchanged outer schema cannot hide wire drift."""
    value = protocol()
    value["canary"][0]["envelope"]["payload"]["seed"] += 1
    with pytest.raises(ValueError, match="envelope"):
        q.bind(value, [])

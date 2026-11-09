"""REQ-REPORT-8950; SCENARIO-REPORT-8950-CONTRACT: independent shard B custody."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock

import pytest

from carnot.reporting import matched_corpus_a_8949 as a
from carnot.reporting import matched_corpus_b_8950 as b
from carnot.reporting import matched_corpus_runner_8949 as runner
from carnot.reporting import graph_source_8931 as source
from carnot.reporting.current_work_receipt import atomic_json
from test_matched_depth_control_8948 import response


@pytest.fixture(scope="module")
def units():
    """REQ-REPORT-8950: authenticate the frozen population without querying a model."""
    evidence = b.authenticate(b.ROOT)
    assert evidence["failures"] == []
    return b.ordered_units(evidence["protocol"])


def test_independent_roster_and_labels(units):
    """REQ-REPORT-8950: preserve A defaults and reject foreign shard roles."""
    protocol = b.authenticate(b.ROOT)["protocol"]
    assert not {u["graph_hash"] for u in units} & {
        u["graph_hash"] for u in a.ordered_units(protocol)
    }
    assert len(units) == 160 and all(u["shard"] == "B" for u in units)
    with pytest.raises(ValueError, match="shard_roster"):
        b.ordered_units(dict(graph_manifest=[]))
    rows = [source.label_response(u, response(u["expected"][0])) for u in units]
    reduced = b.reduce(units, rows)
    assert reduced["split_counts"] == dict(train=80, calibration=40, test=40)
    assert reduced["labels_verified"] and reduced["usable_signals"]
    assert all(r["shard_id"] == "B" for r in reduced["graph_identity_rows"])
    assert all(
        set(r) == set(source.FEATURE_SCHEMA["learned_features"]) for r in reduced["detector_view"]
    )


def test_b_build_and_coverage_binding(units, tmp_path, monkeypatch):
    """SCENARIO-REPORT-8950-CONTRACT: B readiness binds B checks and exact receipts."""
    work = runner.freeze(b.ROOT, tmp_path / "raw", True, producer=b)
    raw = Path(work["raw"])
    validation = dict(receipts=[], coverage={})
    atomic_json(raw / "work.json", work)
    atomic_json(raw / "validation.json", validation)
    assert b.build(work, validation, False)["verdict_class"] == "disqualified"
    receipts = [dict(name=n, passed=True, exit_code=0) for n in a.methods.CHECKS]
    monkeypatch.setattr(a, "verify_measured", lambda coverage, owned: owned == b.OWNED)
    assert b.qualified(dict(receipts=receipts, coverage={}))
    assert not b.qualified(validation)
    monkeypatch.setattr(b, "qualified", lambda _: True)
    work["runtime"]["rows"] = [source.label_response(u, response(u["expected"][0])) for u in units]
    work["runtime"]["live"] = True
    monkeypatch.setattr(b, "raw_complete", lambda _: True)
    value = b.build(work, validation, False)
    assert (value["experiment_id"], value["task_id"], value["shard_id"]) == (8950, b.TASK, "B")
    assert value["shard_ready_score"] == 1 and value["verdict_class"] == "circular_positive"
    assert value["random_seed"] == 765050
    assert b.CLI in value["reproduction_commands"][0]
    assert "shard B" in value["methodology_note"]
    work["authentication"]["failures"] = [dict(failure_kind="missing_file")]
    assert b.build(work, validation, False)["verdict_class"] == "blocked"


def test_private_cli_replay_and_tamper(tmp_path):
    """REQ-REPORT-8950: direct private CLI from cold cwd rejects rehashed headline drift."""
    output = tmp_path / (b.NAME + ".json")
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    command = [sys.executable, "-u", str(b.ROOT / b.CLI)]

    def child(args):
        return subprocess.run(
            command + args, env=env, cwd="/tmp", capture_output=True, text=True, timeout=90
        )

    result = child(["--date", "20261109", "--fixture-e2e", str(output)])
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(output.read_text())
    assert value["shard_id"] == "B" and value["shard_ready_score"] == 0
    assert child(["--cold-replay", str(output)]).returncode == 0
    value["completed_count"] = 160
    value["reproducibility_checksum"] = a.pilot.checksum(value)
    atomic_json(output, value)
    assert child(["--cold-replay", str(output)]).returncode == 1
    assert child(["--date", "wrong"]).returncode == 2


def test_live_dispatch_and_owned_lease(units, tmp_path, monkeypatch):
    """REQ-REPORT-8950: B dispatch owns its lease and never reads an A result."""
    from carnot.inference import matched_corpus_runtime_8949 as runtime

    def fake_run(scheduled, raw, *, task_id):
        assert task_id == b.TASK and scheduled == units
        result = dict(
            rows=[source.censored(u, "private") for u in scheduled],
            counts={},
            checks=[],
            live=False,
        )
        atomic_json(raw / "runtime.json", result)
        return result

    original = runtime.run
    monkeypatch.setattr(runtime, "run", fake_run)
    assert not runner.freeze(b.ROOT, tmp_path / "raw", False, producer=b)["runtime"]["live"]
    monkeypatch.setattr(runtime, "run", original)
    gpu = dict(index=0, uuid="GPU-0", used_mb=0, free_mb=24000)
    monkeypatch.setattr(
        runtime, "preflight", lambda _: dict(model=dict(model_path="private"), gpu=gpu)
    )
    acquire = Mock(side_effect=runtime.LeaseError("private_busy"))
    monkeypatch.setattr(runtime.GpuLease, "acquire", acquire)
    result = runtime.run(units, tmp_path / "lease", task_id=b.TASK)
    assert acquire.call_args.kwargs["task_id"] == b.TASK
    assert result["checks"][0]["upstream"] == b.TASK
    assert result["counts"]["model_loads_attempted"] == 0

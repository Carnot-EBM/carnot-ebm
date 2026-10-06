"""REQ-VERIFY-8211 / REQ-REPORT-8211: original causal memory and direct CLI."""

from copy import deepcopy
import json
import os
from pathlib import Path

import pytest

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.verify import calibrated_memory_trajectory_8211 as e


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    """REQ-VERIFY-8211: measure natural bytes only in private scratch."""
    raw = tmp_path_factory.mktemp("trajectory-8211")
    preflight = raw / "private-preflight"
    preflight.mkdir()
    atomic_json(preflight / "test_probe.json", dict(scope="private_test_preflight", writable=True))
    os.environ.setdefault("CARNOT_8211_PREFLIGHT", str(preflight))
    logs = raw / "logs"
    logs.mkdir()
    worker_log = logs / "measurement.stdout"
    worker_log.write_text("worker still running\n")
    measured = e.measure(e.ROOT, raw)
    with worker_log.open("a") as stream:
        stream.write("worker normal exit\n")
    assert all(r["path"] != str(worker_log) for r in measured["raw_shard_hashes"])
    return measured, raw


def test_original_protocol_rows_and_budget(work):
    """REQ-VERIFY-8211: chronology and repeated seeds retain original masks."""
    measured, raw = work
    value = e.build(measured, raw, [dict(passed=True, normal_exit=True)], fixture=True)
    assert value["input_ready"] == value["learning_trajectory_ready_score"] == 1
    assert value["verdict_class"] == "null"
    assert value["continuous_self_learning_task"] and value["no_model_weight_mutation"]
    assert value["MODEL_SPECS"] == [] and not any(value["model_invocation_counts"].values())
    assert (
        value["independent_generalization_score"]
        == value["generalized_learning_benefit_score"]
        == 0
    )
    assert value["intended_count"] == 192
    assert value["completed_count"] + value["excluded_count"] + value["censored_count"] == 192
    assert len(value["state_manifest"]) == 20
    assert len(value["issued_rows"]) == 20 * 256 * 5
    assert {r["seed"] for r in value["issued_rows"]} == set(range(101, 121))
    assert {r["arm"] for r in value["issued_rows"]} == set(e.m.NAMES.values())
    assert len(value["release_rows"]) == 20 * 236
    assert all(r["release_slot"] == r["label_slot"] + 20 for r in value["release_rows"])
    assert all(r["install_slot"] <= 224 for r in value["admission_rows"])
    assert all(
        r["overlap_kind"] == "continuous_gaussian_inner_product"
        for r in value["support_overlap_rows"]
    )
    assert all(r["passed"] for r in value["restart_state_hashes"])
    assert [r["actual_exit"] for r in value["child_exit_rows"]] == [0, 73, 0]
    failed = e.build(measured, raw, [dict(passed=False)], fixture=True)
    assert (
        failed["verdict_class"] == "disqualified" and not failed["learning_trajectory_ready_score"]
    )
    assert value["hardware_receipt"]["update_operations"] > 0
    assert value["ended_monotonic_ns"] >= value["started_monotonic_ns"]


def test_future_labels_duplicates_late_delivery_and_rollback(tmp_path):
    """SCENARIO-VERIFY-8211-CAUSAL: future targets cannot change earlier issues."""
    rows, labels = e.m.fixture("learnable")
    path = tmp_path / "labels.json"
    evaluator = [dict(r, y=y) for r, y in zip(rows, labels, strict=True)]
    atomic_json(path, dict(rows=evaluator))
    baseline = e.run_seed(rows, path, 101, tmp_path / "baseline")
    evaluator[-1]["y"] = 1 - evaluator[-1]["y"]
    evaluator[159]["y"] = 1 - evaluator[159]["y"]
    atomic_json(path, dict(rows=evaluator))
    changed = e.run_seed(rows, path, 101, tmp_path / "changed")
    assert baseline["issued"][:180] == changed["issued"][:180]
    delivery = e.ReleasedLabels(path, rows, tmp_path / "delivery.jsonl")
    with pytest.raises(ValueError, match="unsealed_release"):
        delivery[0]
    delivery.state = dict(phase="release", cursor=21)
    assert delivery[0] in [0, 1]
    with pytest.raises(ValueError, match="duplicate_delivery"):
        delivery[0]
    delivery.state["cursor"] = 23
    with pytest.raises(ValueError, match="unsealed_release"):
        delivery[1]
    delivery.state["cursor"] = 22
    assert delivery[1] in [0, 1]
    with pytest.raises(ValueError, match="rollback"):
        e.run_seed(rows, path, 101, tmp_path / "baseline", state=e.m.genesis(rows, 101))
    corrupted = deepcopy(baseline)
    corrupted["pending"] = []
    with pytest.raises(ValueError, match="rollback"):
        e.run_seed(rows, path, 101, tmp_path / "baseline", state=corrupted)


def test_external_block_and_absent_gate(tmp_path):
    """SCENARIO-VERIFY-8211-REPLAY: missing input is distinct from zero readiness."""
    work = e.measure(tmp_path, tmp_path / "raw", fixture=True)
    value = e.build(work, tmp_path / "raw", [dict(passed=True)], fixture=True)
    assert value["verdict_class"] == "blocked" and value["rows"] == []
    assert not value["learning_trajectory_ready_score"]
    upstream = json.loads((e.ROOT / e.UPSTREAM).read_text())
    upstream.pop("stream_input_ready_score")
    atomic_json(tmp_path / e.UPSTREAM, upstream)
    work = e.measure(tmp_path, tmp_path / "missing_field", fixture=True)
    check = next(r for r in work["gate_check_summary"] if not r["passed"])
    assert check["artifact_field"] == "stream_input_ready_score" and check["observed"] is None
    upstream["stream_input_ready_score"] = 0
    atomic_json(tmp_path / e.UPSTREAM, upstream)
    assert not e.measure(tmp_path, tmp_path / "zero_field", fixture=True)["input_ready"]


def test_cold_replay_rejects_rehashed_changes(work, tmp_path):
    """SCENARIO-VERIFY-8211-REPLAY: seals alone cannot certify changed measurements."""
    measured, raw = work
    value = e.build(measured, raw, [dict(passed=True)], fixture=True)
    path = tmp_path / (e.NAME + ".json")
    atomic_json(path, value)
    assert e.replay(path)
    for key in [
        "completed_count",
        "learning_trajectory_ready_score",
        "issued_rows",
        "release_rows",
        "support_overlap_rows",
        "trajectory_sha256",
        "experiment_id",
        "rows",
    ]:
        bad = deepcopy(value)
        bad[key] = [] if isinstance(bad[key], list) else "changed"
        bad.pop("reproducibility_checksum")
        bad["reproducibility_checksum"] = canonical_hash(bad)
        atomic_json(path, bad)
        assert not e.replay(path)
    atomic_json(path, dict(value, reproducibility_checksum="wrong"))
    assert not e.replay(path)
    assert not e.replay(tmp_path / "absent.json")
    for key, field in [("code_config_hashes", e.MODULE), ("raw_shard_hashes", None)]:
        bad = deepcopy(value)
        if field:
            bad[key][field] = "wrong"
        else:
            bad[key][0]["sha256"] = "wrong"
        bad.pop("reproducibility_checksum")
        bad["reproducibility_checksum"] = canonical_hash(bad)
        atomic_json(path, bad)
        assert not e.replay(path)


def test_manifest_and_real_cli(work, tmp_path, monkeypatch):
    """SCENARIO-REPORT-8211-CLI: owned paths work outside ambient imports."""
    from carnot.reporting import calibrated_trajectory_execution_8211 as runner
    from carnot.reporting import v709_execution

    measured, raw = work
    value = e.build(measured, raw, [dict(passed=True)], fixture=True)
    path = tmp_path / (e.NAME + ".json")
    atomic_json(path, value)
    specs = runner.manifest(tmp_path, path)
    assert "patch = _exit" in (tmp_path / "coverage.ini").read_text()
    assert "--files" in next(s for s in specs["commands"] if s["name"] == "spec_coverage")["argv"]
    assert specs["repository_health"]["argv"][-2:] == ["tests/python", "-q"]
    cli = ["/usr/bin/env", "-u", "PYTHONPATH"]
    config = os.environ.get("COVERAGE_RCFILE")
    if config:
        cli.append("COVERAGE_PROCESS_START=" + config)
    cli.extend([str(e.ROOT / ".venv/bin/python"), "-u", str(e.ROOT / e.CLI)])
    monkeypatch.setattr(v709_execution, "ROOT", tmp_path)
    for name, args, expected in [
        ("replay", ["--cold-replay", str(path)], 0),
        (
            "blocked",
            [
                "--root",
                str(tmp_path / "absent"),
                "--fixture-output",
                str(tmp_path / "blocked" / path.name),
            ],
            0,
        ),
        (
            "worker",
            [
                "--root",
                str(tmp_path / "absent"),
                "--worker-output",
                str(tmp_path / "worker/measurement.json"),
            ],
            0,
        ),
        ("guard", ["--fixture-output", str(e.ROOT / "results/forbidden.json")], 2),
        ("date", ["--date", "20261005"], 2),
    ]:
        receipt = e.run_check(
            e.ROOT,
            dict(name=name, argv=cli + args, deadline_s=120, expected_exit=expected),
            tmp_path,
            tmp_path / "logs",
        )
        assert receipt["passed"], Path(receipt["stderr_path"]).read_text()
    bad = dict(value, experiment_id=0)
    bad.pop("reproducibility_checksum")
    bad["reproducibility_checksum"] = canonical_hash(bad)
    atomic_json(path, bad)
    receipt = e.run_check(
        e.ROOT,
        dict(
            name="tamper", argv=cli + ["--cold-replay", str(path)], deadline_s=120, expected_exit=1
        ),
        tmp_path,
        tmp_path / "logs",
    )
    assert receipt["passed"]
    parent = tmp_path / "outer"
    parent.mkdir()
    monkeypatch.setenv("CARNOT_8211_COVERAGE_PARENT", str(parent))
    e.copy_child_coverage(raw)
    assert list(parent.glob(".coverage.*"))


def test_replay_rejects_runtime_faults(work, tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8211-REPLAY: faults in independent readers cannot earn readiness."""
    measured, raw = work
    value = e.build(measured, raw, [dict(passed=True)], fixture=True)
    path = tmp_path / (e.NAME + ".json")
    atomic_json(path, value)
    originals = [json.loads(Path(r["state"]["path"]).read_text()) for r in value["state_manifest"]]
    original_journal, original_evidence = e.journal, e.evidence
    with monkeypatch.context() as local:
        local.setattr(
            e, "journal", lambda p: [] if p.name == "issued.jsonl" else original_journal(p)
        )
        assert not e.replay(path)
    with monkeypatch.context() as local:
        local.setattr(e.m, "run", lambda *a, **kw: dict(originals[0], cursor=999))
        assert not e.replay(path)
    with monkeypatch.context() as local:
        local.setattr(e.m, "run", lambda rows, labels, seed: deepcopy(originals[seed - 101]))
        assert e.replay(path)
        local.setattr(
            e, "evidence", lambda *args: dict(original_evidence(*args), hardware_receipt={})
        )
        assert not e.replay(path)
    with monkeypatch.context() as local:
        local.setattr(e.m, "run", lambda rows, labels, seed: deepcopy(originals[seed - 101]))
        local.setattr(
            e.m.engine.historical.LabelVault, "release", lambda *args, **kwargs: dict(y=2)
        )
        assert not e.replay(path)
    with monkeypatch.context() as local:
        local.setattr(e.m, "run", lambda rows, labels, seed: deepcopy(originals[seed - 101]))
        bad = deepcopy(value)
        bad["rows"][-1]["numerator"] += 1
        bad.pop("reproducibility_checksum")
        bad["reproducibility_checksum"] = canonical_hash(bad)
        atomic_json(path, bad)
        assert not e.replay(path)
    stream = tmp_path / "log"
    stream.write_text("changed stdout")
    bad = e.build(
        measured, raw, [dict(passed=True, stdout_path=str(stream), stdout_sha256="wrong")]
    )
    atomic_json(path, bad)
    assert not e.replay(path)

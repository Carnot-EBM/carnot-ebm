"""REQ-VERIFY-8241 / REQ-REPORT-8241: costs and retention need primitive proof."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess

import pytest

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.verify import delayed_benefit_audit_8241 as e


@pytest.fixture
def bundle(tmp_path):
    """SCENARIO-VERIFY-8241-REPLAY: private targets never replace natural rows."""
    rows, path, retained = e.k.fixture(tmp_path / "inputs")
    retained = [
        dict(
            r,
            unit_id="retention-" + r["unit_id"],
            source_cluster_id="retention-" + r["source_cluster_id"],
        )
        for r in retained
    ]
    raw = tmp_path / "seed-101"
    states = [e.k.run(rows, path, 101, raw)]
    data = e.upstream.legacy.evidence(states, rows, retained, tmp_path)
    labels = json.loads(path.read_bytes())["rows"][:64]
    labels = [
        dict(r, unit_id=retained[i]["unit_id"], source_cluster_id=retained[i]["source_cluster_id"])
        for i, r in enumerate(labels)
    ]
    label_path = tmp_path / "retention-labels.json"
    atomic_json(label_path, dict(rows=labels))
    return dict(
        data,
        states=states,
        public=rows,
        retained=retained,
        retention_label_path=str(label_path),
        stream_label_path=str(path),
    )


def cli(tmp_path, *args):
    """SCENARIO-REPORT-8241-CLI: execute the actual script outside the checkout."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    argv = [str(e.ROOT / ".venv/bin/python"), "-u", str(e.ROOT / e.CLI), *map(str, args)]
    if env.get("COVERAGE_RCFILE"):
        argv[1:2] = ["-m", "coverage", "run"]
    print("before private8241 CLI", flush=True)
    result = subprocess.run(
        argv, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=180
    )
    print("after private8241 CLI", result.returncode, flush=True)
    return result


def test_reconstruct_metrics_and_causality(bundle, tmp_path):
    """REQ-VERIFY-8241: every issue and retention head is checked before labels."""
    result = e.reconstruct(bundle, tmp_path / "audit")
    assert len(result["rows"]) == (192 + 64) * 5
    assert (
        result["retention_access"]["predictions_sealed_ns"]
        < result["retention_access"]["labels_opened_ns"]
    )
    assert any(r["status"] == "censored" for r in result["rows"])
    assert any(r["status"] == "excluded" for r in result["rows"])
    stats = e.statistics(result["rows"])
    assert len(stats["per_source_deltas"]) == 192
    assert [r["block_length"] for r in stats["block_bootstrap_diagnostics"]] == [16, 8, 32]
    assert all(r["draws"] == 10000 for r in stats["block_bootstrap_diagnostics"])
    assert e.statistics([])["passed"] is False
    for mutation in ["prediction", "future", "retention", "source", "final", "fit"]:
        changed = deepcopy(bundle)
        if mutation == "prediction":
            changed["states"][0]["frozen"]["issued"][0]["p"] = 0.3
        elif mutation == "future":
            changed["release_log"][0]["release_slot"] = 1
        elif mutation == "retention":
            changed["retention_predictions"][0]["p"] = 0.3
        elif mutation == "source":
            changed["retained"][0]["source_cluster_id"] = changed["public"][0]["source_cluster_id"]
        elif mutation == "final":
            changed["states"][0]["frozen"]["model"] = dict(kind="global", scale=1, intercept=1)
        else:
            event = next(
                r
                for r in changed["states"][0]["global_only"]["events"]
                if r["kind"] == "commit_candidate"
            )
            event["fit_ids"] = [256]
        with pytest.raises(ValueError):
            e.reconstruct(changed, tmp_path / mutation)
    bad = deepcopy(result["rows"])
    bad.append(bad[0])
    with pytest.raises(ValueError):
        e.statistics(bad)


def test_cli_publication_and_tamper(bundle, tmp_path):
    """SCENARIO-REPORT-8241-CLI: null completion and rehashed drift cross real CLI."""
    source = tmp_path / "bundle.json"
    atomic_json(source, bundle)
    output = tmp_path / (e.NAME + ".json")
    result = cli(tmp_path, "--fixture-output", output, "--stream-path", source)
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(output.read_bytes())
    assert value["learning_audit_ready_score"] == 1
    assert value["verdict_class"] in ["null", "circular_positive"]
    assert (
        value["independent_generalization_score"]
        == value["generalized_learning_benefit_score"]
        == 0
    )
    assert value["MODEL_SPECS"] == [] and not any(value["model_invocation_counts"].values())
    assert cli(tmp_path, "--cold-replay", output).returncode == 0
    changed = dict(value, learning_audit_ready_score=9)
    changed.pop("reproducibility_checksum")
    changed["reproducibility_checksum"] = canonical_hash(changed)
    atomic_json(output, changed)
    assert cli(tmp_path, "--cold-replay", output).returncode == 1
    assert not e.replay(tmp_path / "missing")


def test_blocking_worker_and_manifest(tmp_path):
    """REQ-REPORT-8241: missing evidence blocks while owned checks disqualify."""
    output = tmp_path / (e.NAME + ".json")
    assert cli(tmp_path, "--root", tmp_path, "--fixture-output", output).returncode == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked" and value["learning_audit_ready_score"] == 0
    assert value["gate_check_summary"][-1]["observed"] is None
    assert cli(tmp_path, "--cold-replay", output).returncode == 0
    assert cli(tmp_path, "--date", "19000101").returncode == 2
    assert cli(tmp_path, "--fixture-output", e.ROOT / "results/never-write.json").returncode == 2
    worker = tmp_path / "worker/measurement.json"
    assert cli(tmp_path, "--root", tmp_path, "--worker-output", worker).returncode == 0
    work = json.loads(worker.read_bytes())
    assert e.build(work, worker.parent, [dict(passed=False)])["verdict_class"] == "disqualified"
    specs = e.manifest(tmp_path, output)
    assert "--fail-under=100" in specs["commands"][3]["argv"]
    assert "--files" in specs["commands"][-1]["argv"]
    assert specs["repository_health"]["classification"] == "diagnostic"


def test_active_worker_logs_are_sealed_only_after_exit(tmp_path):
    """REQ-REPORT-8241: active stdout is bound by the normal-exit receipt."""
    raw = tmp_path / "running"
    (raw / "logs").mkdir(parents=True)
    log = raw / "logs/measurement.stdout"
    log.write_text("worker still running")
    work = e.measure(tmp_path, raw)
    assert str(log) not in {r["path"] for r in work["raw_shard_hashes"]}


def test_cost_benefit_and_per_seed_safety(tmp_path):
    """REQ-VERIFY-8241: real cost gain passes; extra false accepts fail per seed."""
    rows = []
    for condition, slots in [("later_stream", range(65, 257)), ("retention", range(1, 65))]:
        for slot in slots:
            y = slot % 2
            source = dict(
                slot=slot, unit_id=condition + str(slot), source_cluster_id=condition + str(slot)
            )
            for seed in [101, 102]:
                for arm in e.k.ARMS:
                    p = (0.95 if y else 0.05) if arm == "global_plus_group" else 0.5
                    action = (
                        ("reject" if y else "accept") if arm == "global_plus_group" else "escalate"
                    )
                    rows.append(e.scored(source, dict(p=p, action=action), y, condition, seed, arm))
    stats = e.statistics(rows)
    assert stats["passed"] and stats["retention_passed"]
    assert stats["improved_sources"] == 172
    assert stats["block_bootstrap_diagnostics"][0]["lower_bound"] > 0.02
    work = dict(
        rows=rows,
        owned_failure="",
        checks=[],
        input_ready=1,
        fixture_mode=True,
        duration_s=1,
        clock={},
        refs=[],
        install_exposure_rows=[],
    )
    assert (
        e.build(work, tmp_path, [dict(passed=True)], fixture=True)["verdict_class"]
        == "circular_positive"
    )
    bad = deepcopy(rows)
    row = next(
        r
        for r in bad
        if r["seed"] == 101
        and r["condition"] == "later_stream"
        and r["arm"] == "global_plus_group"
        and r["y"] == 1
    )
    row.update(false_accept=1, action="accept", numerator=5)
    assert not e.statistics(bad)["operands"]["per_seed_safety"]
    with pytest.raises(ValueError):
        e.statistics(rows[1:])


def test_authenticated_natural_bytes_and_external_schema(tmp_path):
    """REQ-VERIFY-8241: natural authentication binds prior negative dispositions."""
    work = dict(checks=[], refs=[])
    data = e.authenticate(e.ROOT, tmp_path / "natural", work)
    assert len(data["states"]) == 20
    assert work["upstream_dispositions"][1]["verdict_class"] == "disqualified"
    target = tmp_path / e.UPSTREAM
    target.parent.mkdir(parents=True)
    atomic_json(target, [])
    with pytest.raises(ValueError, match="input_schema"):
        e.authenticate(tmp_path, tmp_path / "schema", dict(checks=[], refs=[]))
    atomic_json(
        target, dict(experiment_id=8240, task_id=e.upstream.TASK, utility_trajectory_ready_score=0)
    )
    with pytest.raises(ValueError, match="utility_trajectory_ready_score"):
        e.authenticate(tmp_path, tmp_path / "field", dict(checks=[], refs=[]))


def test_owned_failure_and_primitive_tamper(bundle, tmp_path):
    """SCENARIO-VERIFY-8241-REPLAY: owned failures and rehashed primitives fail closed."""
    source = tmp_path / "bundle.json"
    broken = deepcopy(bundle)
    broken["states"][0]["frozen"]["issued"][0]["p"] = 0.3
    atomic_json(source, broken)
    work = e.measure(tmp_path, tmp_path / "bad", fixture=True, stream_path=source)
    assert (
        work["owned_failure"]
        and e.build(work, tmp_path / "bad", [dict(passed=True)])["verdict_class"] == "disqualified"
    )
    atomic_json(source, dict(retention_label_path="/missing", stream_label_path="/missing"))
    work = e.measure(tmp_path, tmp_path / "schema", fixture=True, stream_path=source)
    assert work["input_ready"] == 0
    atomic_json(source, bundle)
    raw = tmp_path / "good"
    work = e.measure(tmp_path, raw, fixture=True, stream_path=source)
    log = tmp_path / "stdout"
    log.write_text("normal exit")
    receipts = [dict(passed=True, stdout_path=str(log), stdout_sha256=e.sha256_file(log))]
    value = e.build(work, raw, receipts, fixture=True)
    output = tmp_path / "candidate.json"
    atomic_json(output, value)
    assert e.replay(output)
    original_value = deepcopy(value)
    changed = deepcopy(value)
    changed["raw_shard_hashes"][0]["sha256"] = "sha256:changed"
    changed.pop("reproducibility_checksum")
    changed["reproducibility_checksum"] = canonical_hash(changed)
    atomic_json(output, changed)
    assert not e.replay(output)
    atomic_json(output, value)
    log.write_text("tampered")
    assert not e.replay(output)
    log.write_text("normal exit")
    changed = dict(value, reproducibility_checksum="bad")
    atomic_json(output, changed)
    assert not e.replay(output)
    atomic_json(output, value)
    original = (raw / "measurement.json").read_bytes()
    work["retention_access"]["labels_opened_ns"] = 0
    atomic_json(raw / "measurement.json", work)
    changed = e.build(work, raw, receipts, fixture=True)
    atomic_json(output, changed)
    assert not e.replay(output)
    (raw / "measurement.json").write_bytes(original)
    atomic_json(output, original_value)
    (raw / "audit_inputs.json").write_text("{}")
    assert not e.replay(output)
    changed = deepcopy(bundle)
    event = next(
        r for r in changed["states"][0]["global_only"]["events"] if r["kind"] == "admit_once"
    )
    event["labels"] = [256]
    with pytest.raises(ValueError, match="future_admission"):
        e.reconstruct(changed, tmp_path / "future")

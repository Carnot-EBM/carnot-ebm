"""REQ-VERIFY-8156 / REQ-REPORT-8156: private independent decision custody."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess

import pytest

from carnot.verify import decision_audit_8156 as e
from carnot.reporting.current_work_receipt import atomic_json


def cli(tmp_path, *args):
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env.pop("PYTHONPATH", None)
    argv = [str(e.ROOT / ".venv/bin/python"), "-u", str(e.ROOT / e.CLI), *map(str, args)]
    if env.get("COVERAGE_RCFILE"):
        argv[1:2] = ["-m", "coverage", "run"]
    print("before private subprocess", flush=True)
    result = subprocess.run(argv, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60)
    print("after private subprocess", result.returncode, flush=True)
    return result


def test_private_cli_and_mutations(tmp_path):
    """SCENARIO-REPORT-8156: genuine script execution, block and cold rejection."""
    path = tmp_path / "experiment_8156_fixture.json"
    result = cli(tmp_path, "--fixture-output", path)
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(path.read_text())
    assert value["decision_audit_ready_score"] == 1
    assert value["h1_development_signal_score"] == 0
    assert value["verdict_class"] == "circular_positive"
    assert value["model_invocation_counts"]["generation_calls_attempted"] == 0
    assert cli(tmp_path, "--cold-replay", path).returncode == 0
    for key, changed in [("completed_count", 1), ("h1_development_signal_score", 1), ("rows", [])]:
        bad = deepcopy(value)
        bad[key] = changed
        atomic_json(path, bad)
        assert cli(tmp_path, "--cold-replay", path).returncode == 1
    atomic_json(path, value)
    work = json.loads(Path(value["measurement_reference"]["path"]).read_text())
    for ref in work["raw_shard_hashes"]:
        primitive = Path(ref["path"])
        original = primitive.read_bytes()
        primitive.write_text("{}")
        assert not e.replay(path)
        primitive.write_bytes(original)
    for receipt in value["validation_receipts"]:
        if receipt.get("log_path"):
            Path(receipt["log_path"]).write_text("tamper")
            assert not e.replay(path)
    blocked = tmp_path / "blocked/experiment_8156_fixture.json"
    assert cli(tmp_path, "--fixture-output", blocked, "--mutation", "block").returncode == 0
    b = json.loads(blocked.read_text())
    assert b["verdict_class"] == "blocked" and b["decision_audit_ready_score"] == 0
    assert e.replay(blocked)
    assert cli(tmp_path, "--fixture-output", e.ROOT / "results/private.json").returncode == 2
    assert cli(tmp_path, "--mutation", "block").returncode == 2
    assert cli(tmp_path, "--date", "19990101").returncode == 2
    assert e.main(["--cold-replay", str(tmp_path / "absent")]) == 1


def test_join_and_clock_controls(tmp_path):
    """SCENARIO-VERIFY-8156: label leakage, absent rows and substituted IDs reject."""
    work = e.measure(tmp_path, tmp_path / "raw", fixture=True)
    data = work["evidence"]
    assert len(e.reduce(data)["rows"]) == 1024
    for mutate in [
        lambda d: d["predictions"][0].update(unit_id="wrong"),
        lambda d: d["predictions"].pop(),
        lambda d: d["targets"][0].update(y=1 - d["targets"][0]["y"]),
        lambda d: d["targets"][0].update(source_cluster_id="wrong"),
        lambda d: d["clock"].update(predictions_sealed_ns=d["clock"]["labels_opened_ns"] + 1),
        lambda d: d["calls"][0].update(human_target=1),
        lambda d: d["predictions"][0].update(p=0.7),
        lambda d: d["calls"].pop(),
    ]:
        bad = deepcopy(data)
        mutate(bad)
        with pytest.raises((ValueError, KeyError)):
            e.reduce(bad)
    receipt = [dict(passed=False)]
    result = e.build(work, tmp_path / "raw", receipt, fixture=True)
    assert result["verdict_class"] == "disqualified" and result["decision_audit_ready_score"] == 0
    bad = deepcopy(work)
    bad["evidence"]["predictions"][0]["p"] = 0.7
    with pytest.raises(ValueError):
        e.build(bad, tmp_path / "raw", [dict(passed=True)], fixture=True)


def test_statistics_and_detection():
    """REQ-VERIFY-8156: known effects qualify the real paired source reducer."""
    rows = []
    for i in range(96):
        for arm in e.ARMS:
            p = 0.9 if i % 2 else 0.01
            if arm == "additive_cubic":
                p = 0.3
            rows.append(
                e.scored(
                    dict(
                        unit_id=str(i),
                        source_cluster_id=str(i),
                        arm=arm,
                        status="completed",
                        exclusion_reason=None,
                        p=p,
                    ),
                    i % 2,
                )
            )
    stats = e.statistics(rows)
    assert stats["h1_development_signal_score"] == 1
    assert stats["paired_intervals"]["valid_draws"] == 10000
    assert stats["paired_intervals"]["lower_one_sided_975"] > 0.02
    rows[0] = e.scored(dict(rows[0], status="excluded", exclusion_reason="missing", p=None), 0)
    assert e.statistics(rows)["h1_development_signal_score"] == 0
    assert e.statistics([])["paired_intervals"]["valid_draws"] == 0
    for p, expected in [
        (0.1, "escalate"),
        (0.5, "escalate"),
        (0.09, "accept"),
        (0.51, "reject"),
        (None, "escalate"),
    ]:
        assert (
            e.scored(
                dict(
                    unit_id="a",
                    source_cluster_id="s",
                    arm="x",
                    p=p,
                    status="completed",
                    exclusion_reason=None,
                ),
                1,
            )["action"]
            == expected
        )


def test_natural_private_custody_and_owned_runner(tmp_path, monkeypatch):
    """REQ-REPORT-8156: owned CPU routes and receipts use only private output."""
    specs = e.manifest(tmp_path, tmp_path / "candidate.json")
    assert specs["commands"][0]["deadline_s"] == 600
    work = e.measure(e.ROOT, tmp_path / "natural")
    assert not [c for c in work["checks"] if not c["passed"]]
    assert e.reduce(work["evidence"])["eligible_count"] >= 96
    for mutation in [
        lambda d: d["roster"].pop(),
        lambda d: d["roster"][1].update(source_cluster_id=d["roster"][0]["source_cluster_id"]),
        lambda d: d["calls"][0].update(source_bytes=b"wrong".hex()),
        lambda d: d["calls"][0].update(source_cluster_id="wrong"),
    ]:
        bad = deepcopy(work["evidence"])
        mutation(bad)
        with pytest.raises(ValueError):
            e.reduce(bad)
    absent = e.measure(tmp_path, tmp_path / "missing")
    assert absent["checks"][-1]["check"] == "upstream_exists"
    blocked = e.build(absent, tmp_path / "missing", [dict(passed=True)])
    assert blocked["honest_verdict"] == "complete_blocked_upstream_exists"
    ref = dict(path=str(tmp_path / "absent.json"), sha256="bad")
    with pytest.raises(ValueError, match="input_sha256"):
        e.bind(ref, tmp_path, [])
    monkeypatch.setattr(e, "PIN", "bad")
    wrong = e.measure(e.ROOT, tmp_path / "hash")
    assert wrong["checks"][-1]["check"] == "input_sha256"
    monkeypatch.undo()
    value = e.build(work, tmp_path / "natural", [dict(passed=True)])
    assert value["verdict_class"] == "null"
    assert value["failed_count"] == 26 and value["excluded_count"] == 4
    output = tmp_path / "experiment_8156_natural.json"
    atomic_json(output, value)
    assert e.replay(output)
    log = tmp_path / "receipt.log"
    log.write_text("original")
    from carnot.reporting.current_work_receipt import sha256_file

    value["validation_receipts"] = [
        dict(passed=True, log_path=str(log), log_sha256=sha256_file(log))
    ]
    atomic_json(output, value)
    assert e.replay(output)
    log.write_text("changed")
    assert not e.replay(output)
    monkeypatch.setattr(e, "measure", lambda *a, **k: deepcopy(work))

    def fake_manifest(private, candidate):
        private.joinpath("coverage.json").write_text("{}")
        return dict(
            commands=[dict(name="owned")],
            repository_health=dict(name="health"),
            terminal_commands=[],
        )

    monkeypatch.setattr(e, "manifest", fake_manifest)
    monkeypatch.setattr(e.producer, "checked", lambda *a: dict(passed=True))
    monkeypatch.setattr(e.execution, "publish", lambda *a: None)
    assert e.main(["--output", str(output)]) == 0


def test_reconstruction_and_malformed_boundary(tmp_path, monkeypatch):
    """REQ-VERIFY-8156: whole-source exclusions and malformed upstreams survive."""
    w = e.measure(tmp_path, tmp_path / "fixture", fixture=True)
    data = w["evidence"]
    with pytest.raises(ValueError, match="original_prediction_rows"):
        e.reconstruct([], data["heads"])
    monkeypatch.setattr(
        e.producer.fit.lexical, "extract", lambda r: dict(values=None, abstention="missing")
    )
    rebuilt = e.reconstruct(data["calls"], data["heads"])
    assert all(r["status"] == "excluded" for r in rebuilt)
    monkeypatch.undo()
    monkeypatch.setattr(e.producer.fit, "read_bound_sidecar", lambda *a: {})
    bad = e.measure(e.ROOT, tmp_path / "malformed")
    assert bad["checks"][-1]["check"] == "input_custody"
    p = tmp_path / "garbled.json"
    p.write_text("{bad")
    assert not e.replay(p)


def test_original_source_identity(tmp_path):
    """REQ-VERIFY-8156: join original source metadata, not only evaluator copies."""
    work = e.measure(tmp_path, tmp_path / "identity", fixture=True)
    data = deepcopy(work["evidence"])
    data["roster"][0]["source_id"] = "wrong"
    data["targets"][0]["source_id"] = "wrong"
    with pytest.raises(ValueError, match="original_human_source"):
        e.reduce(data)
    bad = deepcopy(work["evidence"])
    bad["public"][0]["source_bytes"] = b"wrong".hex()
    with pytest.raises(ValueError, match="original_source_join"):
        e.reduce(bad)

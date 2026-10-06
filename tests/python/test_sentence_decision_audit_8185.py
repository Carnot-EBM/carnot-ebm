"""REQ-VERIFY-8185 / REQ-REPORT-8185: private source audit qualification."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess

import pytest

from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.verify import sentence_decision_audit_8185 as e


def cli(tmp_path, *args):
    env = dict(os.environ, PYTHONUNBUFFERED="1", JAX_PLATFORMS="cpu")
    env.pop("PYTHONPATH", None)
    argv = [str(e.ROOT / ".venv/bin/python"), "-u", str(e.ROOT / e.CLI), *map(str, args)]
    if env.get("COVERAGE_RCFILE"):
        argv[1:2] = ["-m", "coverage", "run"]
    print("before private subprocess", flush=True)
    result = subprocess.run(argv, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60)
    print("after private subprocess", result.returncode, flush=True)
    return result


def test_private_cli_and_cold_tamper(tmp_path):
    """SCENARIO-REPORT-8185: external execution rejects forged headlines."""
    path = tmp_path / "experiment_8185_fixture.json"
    result = cli(tmp_path, "--fixture-output", path)
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(path.read_text())
    assert value["decision_audit_ready_score"] == 1
    assert value["h1_development_signal_score"] == 0
    assert value["verdict_class"] == "circular_positive"
    assert value["MODEL_SPECS"] == []
    assert all(v == 0 for v in value["model_invocation_counts"].values())
    assert cli(tmp_path, "--cold-replay", path).returncode == 0
    for key, changed in [("completed_count", 1), ("h1_development_signal_score", 1), ("rows", [])]:
        bad = deepcopy(value)
        bad[key] = changed
        atomic_json(path, bad)
        assert cli(tmp_path, "--cold-replay", path).returncode == 1
    atomic_json(path, value)
    for ref in value["raw_shard_hashes"]:
        primitive = Path(ref["path"])
        original = primitive.read_bytes()
        primitive.write_text("{}")
        assert not e.replay(path)
        primitive.write_bytes(original)
    blocked = tmp_path / "blocked/experiment_8185_fixture.json"
    assert cli(tmp_path, "--fixture-output", blocked, "--mutation", "block").returncode == 0
    b = json.loads(blocked.read_text())
    assert b["verdict_class"] == "blocked" and b["decision_audit_ready_score"] == 0
    assert e.replay(blocked)
    assert cli(tmp_path, "--fixture-output", e.ROOT / "results/private.json").returncode == 2
    assert cli(tmp_path, "--mutation", "block").returncode == 2
    assert cli(tmp_path, "--date", "19990101").returncode == 2


def test_original_labels_and_source_rows(tmp_path):
    """SCENARIO-VERIFY-8185: mismatched rows and wrong labels fail independently."""
    work = e.measure(tmp_path, tmp_path / "raw", fixture=True)
    data = work["evidence"]
    reduced = e.reduce(data)
    assert len(reduced["rows"]) == 1024
    assert len(reduced["per_source_results"]) == 128
    assert reduced["equivalent_logistic_parity"]["passed"]
    for mutate in [
        lambda d: d["predictions"][0].update(p=0.7),
        lambda d: d["predictions"].pop(),
        lambda d: d["targets"][0].update(y=1 - d["targets"][0]["y"]),
        lambda d: d["targets"][0].update(source_cluster_id="wrong"),
        lambda d: d["targets"].pop(),
        lambda d: d["capture"]["slots"][1].update(
            source_cluster_id=d["capture"]["slots"][0]["source_cluster_id"]
        ),
        lambda d: d["original_response_records"][0].update(source_id="wrong"),
        lambda d: d["capture"]["calls"][0].update(source_cluster_id="wrong"),
        lambda d: d["capture"]["slots"][0].update(answer_bytes=b"wrong".hex()),
        lambda d: d["clock"].update(labels_opened_ns=0),
    ]:
        bad = deepcopy(data)
        mutate(bad)
        with pytest.raises((ValueError, KeyError)):
            e.reduce(bad)
    disqualified = e.build(work, tmp_path / "raw", [dict(passed=False)])
    assert disqualified["verdict_class"] == "disqualified"
    assert disqualified["decision_audit_ready_score"] == 0


def test_frozen_primary_and_all_nulls():
    """REQ-VERIFY-8185: source bootstrap, support and missing escalation are fixed."""
    rows = []
    for i in range(128):
        for arm in e.ARMS:
            p = 0.9 if i % 2 else 0.01
            action = "reject" if i % 2 else "accept"
            if arm == "radial16":
                p, action = 0.3, "escalate"
            rows.append(
                e.score(
                    dict(
                        unit_id=str(i),
                        source_cluster_id=str(i),
                        arm=arm,
                        status="completed",
                        exclusion_reason=None,
                        p=p,
                        action=action,
                    ),
                    i % 2,
                )
            )
    result = e.statistics(rows)
    assert result["h1_development_signal_score"] == 1
    assert result["paired_intervals"]["all_slot"]["valid_draws"] == 10000
    assert result["paired_intervals"]["all_slot"]["lower_one_sided_975"] > 0.02
    assert result["all_slot_metrics"][0]["count"] == 128
    nulls = [
        e.score(dict(r, p=None, status="failed", exclusion_reason="transport"), r["y"])
        for r in rows
    ]
    result = e.statistics(nulls)
    assert not result["support_sufficient"] and result["h1_development_signal_score"] == 0
    assert all(r["numerator"] == 0.5 for r in nulls)
    assert result["paired_intervals"]["complete_case"]["valid_draws"] == 0
    assert e.statistics([])["eligible_count"] == 0
    assert e.score(dict(rows[0]), None)["denominator"] == 0
    assert e.score(dict(rows[0], p=None), None)["numerator"] == 0.5
    assert e.CONFIG["seed"] == 7078185


def test_natural_custody_and_owned_execution(tmp_path, monkeypatch):
    """REQ-REPORT-8185: authenticate real inputs while keeping all outputs private."""
    specs = e.manifest(tmp_path, tmp_path / "candidate.json")
    assert specs["commands"][0]["deadline_s"] == 180
    assert specs["repository_health"]["argv"][-2:] == ["tests/python", "-q"]
    work = e.measure(e.ROOT, tmp_path / "natural")
    assert work["evidence"], work["checks"][-1]
    assert all(c["passed"] for c in work["checks"])
    reduced = e.reduce(work["evidence"])
    assert reduced["eligible_count"] == 97
    assert len(reduced["per_source_results"]) == 128
    assert all(r["count"] == 128 for r in reduced["all_slot_metrics"])
    assert reduced["paired_intervals"]["all_slot"]["valid_draws"] == 10000
    assert reduced["equivalent_logistic_parity"]["passed"]
    value = e.build(work, tmp_path / "natural", [dict(passed=True)])
    output = tmp_path / "experiment_8185_natural.json"
    atomic_json(output, value)
    assert e.replay(output)
    log = tmp_path / "receipt.log"
    log.write_text("original")
    value["validation_receipts"] = [
        dict(passed=True, log_path=str(log), log_sha256=sha256_file(log))
    ]
    atomic_json(output, value)
    assert e.replay(output)
    log.write_text("changed")
    assert not e.replay(output)
    missing = e.measure(tmp_path, tmp_path / "missing")
    assert missing["checks"][-1]["check"] == "upstream_exists"
    blocked = e.build(missing, tmp_path / "missing", [dict(passed=True)])
    assert blocked["honest_verdict"] == "complete_blocked_upstream_exists"
    monkeypatch.setattr(e, "PIN", "wrong")
    wrong = e.measure(e.ROOT, tmp_path / "hash")
    assert wrong["checks"][-1]["check"] == "input_sha256"
    monkeypatch.undo()
    monkeypatch.setattr(e, "measure", lambda *a, **k: deepcopy(work))

    def fake_manifest(private, candidate):
        private.joinpath("coverage.json").write_text("{}")
        return dict(
            commands=[dict(name="owned")],
            repository_health=dict(name="health"),
            terminal_commands=[],
        )

    monkeypatch.setattr(e, "manifest", fake_manifest)
    monkeypatch.setattr(e.base.producer, "checked", lambda *a: dict(passed=True))
    monkeypatch.setattr(e.execution, "publish", lambda *a: None)
    assert e.main(["--output", str(output)]) == 0


def test_missing_masks_parity_and_rehashed_drift(tmp_path, monkeypatch):
    """SCENARIO-VERIFY-8185: missing transport is null; forged hashes do not repair it."""
    work = e.measure(tmp_path, tmp_path / "raw", fixture=True)
    data = deepcopy(work["evidence"])
    data["capture"]["calls"] = []
    data["predictions"] = e.producer.predict(
        e.producer.reduce(data["capture"]["slots"], [], data["capture"]["plan"]["baseline"])[
            "feature_rows"
        ],
        data["capture"]["plan"]["heads"],
    )
    reduced = e.reduce(data)
    assert reduced["eligible_count"] == 0 and reduced["h1_development_signal_score"] == 0
    assert not reduced["equivalent_logistic_parity"]["passed"]
    assert all(r["numerator"] == 0.5 for r in reduced["rows"])
    assert not e.replay(tmp_path / "absent")
    rows = e.reduce(work["evidence"])["rows"]
    with pytest.raises(ValueError, match="duplicate_source_arm"):
        e.statistics([*rows, rows[0]])
    output = tmp_path / "experiment_8185_fixture.json"
    value = e.build(work, tmp_path / "raw", [dict(passed=True)], fixture=True)
    atomic_json(output, value)
    primitive = next(r for r in work["raw_shard_hashes"] if "independent_reduction" in r["path"])
    p = Path(primitive["path"])
    saved = json.loads(p.read_text())
    saved["h1_development_signal_score"] = 1
    atomic_json(p, saved)
    primitive["sha256"] = sha256_file(p)
    atomic_json(tmp_path / "raw/measurement.json", work)
    value["raw_shard_hashes"] = work["raw_shard_hashes"]
    value["measurement_reference"] = e.reference(tmp_path / "raw/measurement.json")
    atomic_json(output, value)
    assert not e.replay(output)
    monkeypatch.setattr(e.producer, "replay", lambda p: (_ for _ in ()).throw(ValueError("drift")))
    bad = e.measure(e.ROOT, tmp_path / "malformed")
    assert bad["checks"][-1]["check"] == "input_custody"


def test_producer_agreement_cannot_replace_independent_math(tmp_path, monkeypatch):
    """REQ-VERIFY-8185: agreeing forged producer rows still fail scalar equations."""
    data = e.measure(tmp_path, tmp_path / "raw", fixture=True)["evidence"]
    data["predictions"][0]["p"] = 0.7
    monkeypatch.setattr(e.producer, "predict", lambda *a: deepcopy(data["predictions"]))
    with pytest.raises(ValueError, match="independent_prediction_drift"):
        e.reduce(data)

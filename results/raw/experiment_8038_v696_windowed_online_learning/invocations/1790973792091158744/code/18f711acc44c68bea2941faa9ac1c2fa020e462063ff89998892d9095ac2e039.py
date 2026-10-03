"""REQ-REPORT-8038: causal replay, equal budgets and durable publication."""

import copy
import json
import os
from pathlib import Path
import sqlite3
import subprocess

import numpy as np
import pytest

from carnot import experiment_8038_v696_windowed_online_learning as e
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.verify import windowed_online_8038 as w
from test_causal_online_8025 import fixture


def test_selection():
    """SCENARIO-REPORT-8038-CAUSAL: window support never uses outcome priorities."""
    rows = [dict(family_id=str(i), origin_slot=i, due_slot=i + 20) for i in range(128)]
    for arm in w.ARMS:
        pool, chosen = w.select(rows, arm, 101, 7)
        assert len(chosen) == (0 if arm == "frozen_no_write" else 4)
        assert len({r["family_id"] for r in chosen}) == len(chosen)
        assert all(r in pool for r in chosen)
        assert w.select(rows, arm, 101, 7) == (pool, chosen)
    assert w.select(rows, "recent64", 101, 7)[0] == rows[-64:]
    assert w.select(rows, "newest16", 101, 7)[1] == w.old.select(rows[-16:], "uniform", 101)
    dirty = [dict(r, brier=999, actual_cost=999, y=1) for r in rows]
    assert [r["family_id"] for r in w.select(dirty, "recent64", 101, 7)[1]] == [
        r["family_id"] for r in w.select(rows, "recent64", 101, 7)[1]
    ]


def test_trajectory(tmp_path):
    """SCENARIO-REPORT-8038-CAUSAL: exclusions preserve delay and exact budgets."""
    data = fixture(96)
    data["labels"]["3"] = None
    data["sources"][5]["public_eligible"] = False
    value = w.measure(data, tmp_path / "first")
    assert w.reduce(tmp_path / "first") == value
    assert len(value["issued_prediction_rows"]) == 4 * 96
    assert all(
        r["actual_gradient_count"] == (0 if r["arm"] == "frozen_no_write" else 16)
        for r in value["update_budget_rows"]
    )
    assert value["sample_size_budget"]["censored"] == 20
    assert value["sample_size_budget"]["excluded"] == 2
    assert value["sparse_dense_max_error"] <= 1e-10
    assert all(r["origin_slot"] + 20 <= r["slot"] for r in value["gradient_rows"])
    assert all(r["family_id"] != "3" for r in value["gradient_rows"])
    frozen = [r for r in value["issued_prediction_rows"] if r["arm"] == "frozen_no_write"]
    assert len({r["head_hash"] for r in frozen}) == 1
    changed = copy.deepcopy(data)
    changed["labels"]["60"] ^= 1
    other = w.measure(changed, tmp_path / "other")
    assert [
        (r["probability"], r["head_hash"])
        for r in value["issued_prediction_rows"]
        if r["slot"] <= 80
    ] == [
        (r["probability"], r["head_hash"])
        for r in other["issued_prediction_rows"]
        if r["slot"] <= 80
    ]
    db = sqlite3.connect(tmp_path / "first" / "ledger.sqlite")
    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            "INSERT INTO events(kind,identity,payload) SELECT kind,identity,payload FROM events LIMIT 1"
        )
    db.close()
    assert value["journal_exactly_once"]
    assert w.controls()["passed"]


def test_deadline_and_tamper(tmp_path):
    """REQ-REPORT-8038: failed numerical or durable evidence cannot qualify."""
    with pytest.raises(TimeoutError, match="numerical_budget"):
        w.measure(fixture(), tmp_path / "expired", budget_s=-1)
    raw = tmp_path / "tamper"
    w.measure(fixture(), raw)
    db = sqlite3.connect(raw / "ledger.sqlite")
    seq, payload = db.execute(
        "SELECT seq,payload FROM events WHERE kind='issue' LIMIT 1"
    ).fetchone()
    row = json.loads(payload)
    row["probability"] = 0.9
    db.execute("UPDATE events SET payload=? WHERE seq=?", (json.dumps(row), seq))
    db.commit()
    db.close()
    with pytest.raises(ValueError):
        w.reduce(raw)


def cli(tmp_path, *args):
    """The real script runs outside the checkout with private guarded output."""
    command = [str(e.ROOT / ".venv/bin/python"), "-u", str(e.ROOT / e.SCRIPT), *map(str, args)]
    config = os.environ.get("CARNOT_8038_COVERAGE_CONFIG")
    if config:
        command[1:2] = ["-m", "coverage", "run", "--rcfile=" + config]
    env = dict(
        os.environ,
        PYTHONUNBUFFERED="1",
        JAX_PLATFORMS="cpu",
        CARNOT_EXPERIMENT_ARTIFACT_ROOT=str(tmp_path),
    )
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        command, cwd=tmp_path, env=env, text=True, capture_output=True, timeout=60
    )
    return result


def test_private_cli(tmp_path):
    """SCENARIO-REPORT-8038-PUBLICATION: actual CLI and consumers bind final bytes."""
    src = tmp_path / "fixture.json"
    atomic_json(src, fixture())
    result = cli(tmp_path, "--fixture-input", src, "--validation-worker")
    assert result.returncode == 0, result.stdout + result.stderr
    output = tmp_path / (e.NAME + ".json")
    value = json.loads(output.read_text())
    assert value["verdict_class"] == "circular_positive"
    assert value["learning_trajectory_ready_score"] == 0
    assert cli(tmp_path, "--cold-replay", output).returncode == 0
    assert e.terminal(output)["passed"]
    value["rows"][0]["probability"] = 0.999
    atomic_json(output, value)
    bad = cli(tmp_path, "--cold-replay", output)
    assert bad.returncode == 1 and "reduction_drift" in bad.stdout
    assert cli(tmp_path, "--date", "20261001").returncode == 2


def source_fixture(root):
    """Private byte-bound protocol exercises the same upstream provenance checks."""
    data = fixture(256)
    for r in data["sources"]:
        r.update(source_bytes=("source-" + r["family_id"]).encode().hex())
    public = root / "public.json"
    target = root / "target.json"
    atomic_json(public, dict(rows=data["sources"]))
    atomic_json(
        target, dict(rows=[dict(family_id=k, eligible_y=v) for k, v in data["labels"].items()])
    )
    protocol = root / "methods.json"
    atomic_json(
        protocol,
        dict(
            methods=e.upstream.METHODS,
            frozen=True,
            head=dict(data["head"], converged=True),
            public=dict(stream=e.reference(public)),
        ),
    )
    path = root / "results" / (e.upstream.NAME + ".json")
    terminal = root / "terminal.json"
    atomic_json(
        path,
        dict(
            experiment_id=8032,
            learning_inputs_ready_score=1,
            flagged_adversarial=False,
            verdict_class="null",
            terminal_validation_sidecar_path=str(terminal),
            methods_reference=e.reference(protocol),
            role_manifests=dict(evaluator=dict(stream=e.reference(target))),
            cited_upstream_artifacts=[e.reference(public)],
            historical_exposure=dict(development_exposed=True),
        ),
    )
    bind(root)
    return data, path


def bind(root):
    """A fixture terminal report authenticates exact bytes rather than a flag."""
    path = root / "results" / (e.upstream.NAME + ".json")
    binding = dict(
        primary_path=str(path),
        primary_sha256=e.sha256_file(path),
        sidecar_path=str(root / "sidecar.json"),
    )
    atomic_json(root / "terminal.json", dict(publication=binding))
    atomic_json(root / "sidecar.json", dict(binding, report=dict(passed=True)))


def test_admission_and_real_target_cli(tmp_path):
    """REQ-REPORT-8038: current frozen protocol supplies the qualified starting head."""
    root = tmp_path / "upstream"
    data, path = source_fixture(root)
    loaded, failures = e.load_inputs(root, tmp_path / "custody")
    assert not failures and len(loaded["sources"]) == 256
    assert loaded["head"] == data["head"]
    assert all(r["passed"] and r["sha256"] for r in loaded["gate_checks"])
    result = cli(tmp_path, "--root", root, "--validation-worker")
    assert result.returncode == 0, result.stdout + result.stderr
    output = tmp_path / (e.NAME + ".json")
    value = json.loads(output.read_text())
    assert value["verdict_class"] == "null"
    assert value["completed_count"] == 256 and value["independent_count"] == 236
    assert {r["actual_gradient_count"] for r in value["update_budget_rows"]} == {0, 56}
    assert value["generalized_learning_benefit_score"] == 0
    assert e.replay(output)["passed"]
    del value["rows"][0]
    atomic_json(output, value)
    with pytest.raises(ValueError, match="reduction_drift"):
        e.replay(output)
    original = json.loads(path.read_text())
    original.pop("learning_inputs_ready_score")
    atomic_json(path, original)
    bind(root)
    _, failed = e.load_inputs(root, tmp_path / "missing-field")
    assert failed[0]["artifact_field"] == "learning_inputs_ready_score"
    assert failed[0]["observed"] == "MISSING_CONTRACT_FIELD"
    original.pop("methods_reference")
    original["learning_inputs_ready_score"] = 1
    atomic_json(path, original)
    bind(root)
    assert e.load_inputs(root, tmp_path / "missing-contract")[1]
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    result = cli(blocked, "--root", blocked / "absent", "--validation-worker")
    assert result.returncode == 0
    assert json.loads((blocked / (e.NAME + ".json")).read_text())["verdict_class"] == "blocked"


@pytest.mark.parametrize("mode", ["pass", "fail", "absent"])
def test_validation_branches(tmp_path, monkeypatch, mode):
    """SCENARIO-REPORT-8038-PUBLICATION: only complete owned validation sets readiness."""
    data = fixture()
    monkeypatch.setenv("CARNOT_EXPERIMENT_ARTIFACT_ROOT", str(tmp_path))
    monkeypatch.setattr(e, "load_inputs", lambda root, raw: (data, []))
    monkeypatch.setattr(e, "terminal", e.replay)

    def run(root, commands, **kwargs):
        if mode != "absent":
            config = Path(kwargs["extra_env"]["CARNOT_8038_COVERAGE_CONFIG"])
            atomic_json(
                config.parent / "coverage.json",
                dict(
                    files={
                        p: dict(
                            summary=dict(
                                num_statements=1,
                                missing_lines=int(mode == "fail"),
                                covered_lines=int(mode == "pass"),
                            )
                        )
                        for p in e.OWNED
                    }
                ),
            )
        return [
            dict(scope="owned", passed=mode == "pass"),
            dict(scope="repository_health", passed=False),
        ]

    monkeypatch.setattr(e, "run_commands", run)
    assert e.main([]) == 0
    output = tmp_path / (e.NAME + ".json")
    value = json.loads(output.read_text())
    assert value["learning_trajectory_ready_score"] == int(mode == "pass")
    assert bool(value["coverage_statement_counts"]) == (mode != "absent")
    assert not value["repository_health"][0]["passed"]
    assert value["verdict_class"] == ("null" if mode == "pass" else "disqualified")
    if mode == "pass":
        value["acceptance_gate_results"]["owned_checks"] = False
        atomic_json(output, value)
        with pytest.raises(ValueError, match="unsafe_readiness"):
            e.replay(output)


def test_store_restart_and_nonunit_calibration(tmp_path):
    """SCENARIO-REPORT-8038-CAUSAL: restart reuses commits and retains calibrated parity."""
    from carnot.reporting.learning_store_8026 import worker

    data = fixture()
    data["head"]["parameters"] = np.linspace(-0.2, 0.3, 110).tolist()
    data["head"]["calibration"] = [0.2, 1.7]
    value = w.measure(data, tmp_path / "trajectory")
    assert value["sparse_dense_max_error"] <= 1e-10
    source = tmp_path / "worker.json"
    atomic_json(
        source,
        dict(
            head=data["head"],
            arm="recent64",
            seed=101,
            releases=[dict(source=data["sources"][i], family_id=str(i), y=i % 2) for i in range(4)],
            next_source=data["sources"][40],
        ),
    )
    first = worker(source, tmp_path / "store", "none")
    assert worker(source, tmp_path / "store", "none") == first
    assert first["exactly_once"]

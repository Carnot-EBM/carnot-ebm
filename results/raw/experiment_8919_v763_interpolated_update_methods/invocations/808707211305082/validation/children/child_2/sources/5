"""REQ-REPORT-8919 / SCENARIO-REPORT-8919-CONTRACT: causal optimizer controls."""

from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from carnot.memory import interpolated_memory_8919 as opt
from carnot.memory import guarded_memory_8905 as g
from carnot.reporting import interpolated_update_methods_8919 as m
from carnot.reporting.current_work_receipt import atomic_json


def test_scalar_controls_and_invalid_inputs() -> None:
    """REQ-REPORT-8919: test the equations independently of the categorical head."""
    fast, slow = opt.advance(np.array([2.0]), np.array([4.0]), np.array([3.0]))
    np.testing.assert_allclose(fast, [1.7])
    np.testing.assert_allclose(slow, [3.425])
    f, s = opt.advance(np.array([2.0]), np.array([4.0]), np.array([0.0]))
    np.testing.assert_allclose(f, [2])
    np.testing.assert_allclose(s, [3.5])
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError):
            opt.advance(np.array([bad]), np.array([1.0]), np.array([0.0]))
    with pytest.raises(ValueError):
        opt.advance(np.array([1.0]), np.array([1.0, 2.0]), np.array([0.0]))


def test_finite_difference_and_base_reduction() -> None:
    """REQ-REPORT-8919: the gradient is evaluated at the interpolated point."""
    params = m.signed.head(11, "signed", [1, 1])
    slow = deepcopy(params)
    slow["weights"] = (np.asarray(params["weights"]) + 0.2).tolist()
    counts = [5, 2, 1, 3, 0, 1, 0.2, 0.5]
    state = dict(fast=params, slow=slow, update_count=0)
    result, receipt = opt.update(state, counts, "support")
    x = np.asarray(m.signed.vector(params, counts))
    weights = (np.asarray(params["weights"][0]) + np.asarray(slow["weights"][0])) / 2

    def loss(w: np.ndarray) -> float:
        logits = w @ x
        return float(np.logaddexp.reduce(logits) - logits[0])

    eps = 1e-5
    for i in range(3):
        for j in range(9):
            delta = np.zeros_like(weights)
            delta[i, j] = eps
            expected = (loss(weights + delta) - loss(weights - delta)) / (2 * eps)
            assert receipt["gradient"][i][j] == pytest.approx(expected, abs=2e-6)
    assert result["update_count"] == 1 and receipt["parameter_delta"] > 0
    base, _ = m.signed.update(params, counts, "support")
    reduced, _ = opt.update(
        dict(fast=params, slow=params, update_count=0), counts, "support", alpha=1, beta=0
    )
    np.testing.assert_allclose(reduced["slow"]["weights"], base["weights"], atol=2e-7)
    with pytest.raises(ValueError):
        opt.update(state, counts, "future_label")
    with pytest.raises(ValueError):
        opt.update(dict(state, update_count=-1), counts, "support")


def test_release_restart_schema_and_atomicity(tmp_path: Path) -> None:
    """REQ-REPORT-8919: future feedback and mixed checkpoints cannot change state."""
    params = m.signed.head(11, "signed", [1, 1])
    s = g.Store(
        tmp_path / "state.json", params, "interpolated_true", m.anchors(), optimizer=opt.SCHEMA
    )
    for i in range(16):
        value = m.old.world(m.old.graphs()["development"][i % 8], f"a/{i}", m.old.ERRORS[i % 6], i)
        with patch.object(g.symbolic, "decide", side_effect=AssertionError("future")):
            s.predict(f"g/{i}", value, i + 1, i + 9)
        if i == 0:
            before = deepcopy(s.state)
            with pytest.raises(ValueError):
                s.release([("g/0", value)], 8)
            assert s.state == before
    batch = [
        (
            f"g/{i}",
            m.old.world(m.old.graphs()["development"][i % 8], f"a/{i}", m.old.ERRORS[i % 6], i),
        )
        for i in range(16)
    ]
    s.release(batch, 24)
    assert s.state["optimizer"]["update_count"] > 0
    assert s.state["candidate"] == s.state["optimizer"]["slow"]
    resumed = g.Store(s.path, {}, "interpolated_true", m.anchors(), optimizer=opt.SCHEMA)
    assert resumed.state == s.state
    with pytest.raises(ValueError):
        s.release(batch, 24)
    with pytest.raises(ValueError):
        g.Store(s.path, {}, "interpolated_true", m.anchors())
    before = deepcopy(s.state)
    with patch.object(g, "atomic_json", side_effect=OSError("disk")):
        with pytest.raises(OSError):
            s.commit(dict(s.state, cursor=99))
    assert s.state == before
    broken = deepcopy(s.state)
    broken["optimizer"]["fast"]["weights"][0][0][0] = float("nan")
    atomic_json(s.path, broken)
    with pytest.raises(ValueError):
        g.Store(s.path, {}, "interpolated_true", m.anchors(), optimizer=opt.SCHEMA)


@pytest.fixture(scope="module")
def pilot_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """REQ-REPORT-8919: reuse private states to test recovery without rerunning pilots."""
    return tmp_path_factory.mktemp("interpolated")


@pytest.fixture(scope="module")
def measured(pilot_dir: Path) -> dict:
    """REQ-REPORT-8919: execute genuine development arithmetic in private scratch."""
    return m.measure(
        pilot_dir, {str(seed): m.signed.head(11, "signed", [1, 1]) for seed in m.old.SEEDS}
    )


def test_protocol_matched_work_and_null(measured: dict, pilot_dir: Path) -> None:
    """SCENARIO-REPORT-8919-CONTRACT: feasibility cannot open evaluation labels."""
    p = m.protocol()
    assert [s["seed"] for s in p["pilot_manifest"]] == list(range(763100, 763108))
    assert [s["seed"] for s in p["frozen_evaluation_manifest"]] == list(range(763200, 763224))
    assert p["evaluation_plan"]["family_intervals"] == 9 and not p["evaluation_opened"]
    assert len(p["evaluation_plan"]["bounds"]) == 9
    assert len(measured["rows"]) == 6144
    assert all(r["passed"] for r in measured["control_rows"])
    assert measured["numerical_training_calls"] > 0
    assert measured["candidate_ready_score"] in (0, 1)
    changed = deepcopy(measured["rows"])
    for row in changed:
        if row["arm"] == "interpolated_true":
            row["metric"] = 2
    assert m.feasibility(changed, measured["promotion_rows"])["candidate_ready_score"] == 0
    permuted = False
    for plan in p["pilot_manifest"]:
        state = json.loads(
            (pilot_dir / "states" / plan["stream_id"] / "interpolated_permuted.json").read_bytes()
        )
        for block in state["blocks"]:
            training = [i for i, group in enumerate(block["groups"]) if g.split(group) == "train"]
            assert sorted(block["labels"][i] for i in training) == sorted(
                block["targets"][i] for i in training
            )
            assert all(
                block["labels"][i] == block["targets"][i]
                for i, group in enumerate(block["groups"])
                if g.split(group) == "guard"
            )
            permuted |= block["labels"] != block["targets"]
    assert permuted


def test_external_gates_and_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-8919-CONTRACT: absent external inputs terminate without training."""
    from carnot.reporting import interpolated_update_runner_8919 as runner

    raw = tmp_path / "raw"
    raw.mkdir()
    assert runner.preconditions(raw, tmp_path)["diagnostics"][0]["check"] == "missing_file"
    name, (_, field, _) = next(iter(runner.PINS.items()))
    for value, expected in [
        ({}, "missing_field"),
        ({field: 0}, "observed_zero"),
        ({field: 1}, "rejected_producer"),
    ]:
        atomic_json(tmp_path / (name + ".json"), value)
        assert runner.preconditions(raw, tmp_path)["diagnostics"][0]["check"] == expected
    (tmp_path / (name + ".json")).write_text("invalid")
    assert not runner.preconditions(raw, tmp_path)["external_ready"]
    output = tmp_path / (m.NAME + ".json")
    assert (
        runner.main(
            ["--date", "20261108", "--sources", str(tmp_path), "--private-output", str(output)]
        )
        == 0
    )
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked" and value["methods_ready_score"] == 0
    assert runner.main(["--replay", str(output)]) == 0
    value["rows"][0]["metric"] = 0.3
    atomic_json(output, value)
    assert runner.main(["--replay", str(output)]) == 1
    with pytest.raises(SystemExit):
        runner.main(["--date", "20261109"])
    with pytest.raises(SystemExit):
        runner.main(["--private-output", str(m.ROOT / "results" / (m.NAME + ".json"))])


@pytest.mark.parametrize(
    "mode", ["null", "producer", "resources", "measurement", "validation", "terminal"]
)
def test_terminal_paths_and_rehashed_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, measured: dict, mode: str
) -> None:
    """REQ-REPORT-8919: owned failure disqualifies; a valid pilot null is terminal."""
    from types import SimpleNamespace
    from carnot.reporting import interpolated_update_runner_8919 as r
    from carnot.reporting.current_work_receipt import canonical_hash
    from carnot.reporting.v746_retained_evidence import reference

    fit = tmp_path / "fit.json"
    atomic_json(fit, dict(heads=dict(signed={"11": m.signed.head(11, "signed", [1, 1])})))
    pre = dict(
        external_ready=True,
        resources=dict(passed=mode != "resources"),
        references=[reference(fit)],
        diagnostics=[],
    )
    monkeypatch.setattr(r, "preconditions", lambda *args: pre)
    monkeypatch.setattr(r, "PINS", {})
    if mode == "producer":
        source = tmp_path / "experiment_8911_fixture.json"
        atomic_json(source, dict(ready=1, task_id="exp8911-fixture"))
        pre["references"].append(reference(source))
        monkeypatch.setattr(
            r, "PINS", {source.stem: (reference(source)["sha256"], "ready", "exp8911-fixture")}
        )
    validation = dict(
        coverage=dict(complete=mode != "validation", percent=100, files=[]),
        union=dict(passed=True),
        receipts=[],
        raw_references=[],
    )
    log = tmp_path / "terminal.log"
    log.write_text("private validation")
    monkeypatch.setattr(
        r,
        "adapter",
        lambda: SimpleNamespace(
            validate=lambda *args: validation,
            terminal=lambda *args: [
                dict(passed=mode != "terminal", stdout_path=str(log), stderr_path=str(log))
            ],
        ),
    )

    def measurement(*args: object) -> dict:
        if mode == "measurement":
            raise ValueError("owned_failure")
        return deepcopy(measured)

    monkeypatch.setattr(m, "measure", measurement)
    output = tmp_path / (m.NAME + ".json")
    assert r.main(["--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == ("null" if mode in ("null", "producer") else "disqualified")
    assert r.replay(output)["passed"]
    for key, changed in [
        ("methods_ready_score", 1 - value["methods_ready_score"]),
        ("numerical_training_calls", -1),
    ]:
        bad = deepcopy(value)
        bad[key] = changed
        bad.pop("reproducibility_checksum")
        bad["reproducibility_checksum"] = canonical_hash(bad)
        atomic_json(output, bad)
        with pytest.raises(ValueError):
            r.replay(output)


def test_actual_private_cli(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8919-CONTRACT: child coverage and foreign-directory cold replay."""
    import os
    import subprocess
    import sys

    env = dict(os.environ, JAX_PLATFORMS="cpu")
    env.pop("PYTHONPATH", None)
    output = tmp_path / (m.NAME + ".json")
    for arguments, expected in [
        (
            [
                "--date",
                "20261108",
                "--private-output",
                str(output),
                "--sources",
                str(tmp_path / "absent"),
            ],
            0,
        ),
        (["--cold-replay", str(output)], 0),
        (["--cold-replay", str(tmp_path / "absent.json")], 1),
    ]:
        child = subprocess.run(
            [sys.executable, "-u", str(m.ROOT / m.CLI), *arguments],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert child.returncode == expected, child.stdout + child.stderr


def test_successful_custody_and_adapter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-8919: authenticate each producer assertion and its original operand."""
    from carnot.reporting import interpolated_update_runner_8919 as r
    from carnot.reporting.v746_retained_evidence import reference

    source = tmp_path / "experiment_8911_fixture.json"
    operand = tmp_path / "operand"
    operand.write_text("original bytes")
    atomic_json(
        source,
        dict(
            experiment_id=8911,
            task_id="exp8911-fixture",
            milestone="2026.11.762",
            required_checks_passed=True,
            ready=1,
            source_artifact_hashes=[reference(operand)],
            code_config_hashes=[],
            raw_shard_hashes=[],
        ),
    )
    monkeypatch.setattr(
        r, "PINS", {source.stem: (reference(source)["sha256"], "ready", "exp8911-fixture")}
    )
    assert r.preconditions(tmp_path / "raw", tmp_path)["external_ready"]
    operand.write_text("changed")
    assert not r.preconditions(tmp_path / "raw", tmp_path)["external_ready"]
    assert r.adapter().m is m
    monkeypatch.setattr(m, "ROOT", tmp_path)
    store = tmp_path / "python/carnot/memory/guarded_memory_8905.py"
    store.parent.mkdir(parents=True)
    store.write_text("original store")
    ref = reference(store)
    retained = tmp_path / "results/raw" / m.NAME / "original_sources" / ref["sha256"].split(":")[1]
    retained.parent.mkdir(parents=True)
    retained.write_bytes(store.read_bytes())
    store.write_text("extended store")
    value = json.loads(source.read_bytes())
    value["source_artifact_hashes"] = [ref]
    atomic_json(source, value)
    monkeypatch.setattr(
        r, "PINS", {source.stem: (reference(source)["sha256"], "ready", "exp8911-fixture")}
    )
    assert r.preconditions(tmp_path / "retained", tmp_path)["external_ready"]
    with patch.object(r, "stream", return_value={"passed": True}) as mocked:
        assert r.adapter().stream("collect_once", [], tmp_path)["passed"]
        assert mocked.call_args.kwargs["deadline"] == 180


def test_partial_recovery(pilot_dir: Path, measured: dict, tmp_path: Path) -> None:
    """REQ-REPORT-8919: an interrupted stream retains its rows and durable training cost."""
    import shutil
    from carnot.reporting import interpolated_update_runner_8919 as r

    plan = m.protocol()["pilot_manifest"][0]
    source = pilot_dir / (plan["stream_id"] + ".json")
    shutil.copyfile(source, tmp_path / source.name)
    shutil.copytree(
        pilot_dir / "states" / plan["stream_id"], tmp_path / "states" / plan["stream_id"]
    )
    failed = m.protocol()["pilot_manifest"][1]["stream_id"]
    atomic_json(tmp_path / ("attempt-" + failed + ".json"), dict(status="failed"))
    result = r.recover(tmp_path, "owned interruption")
    assert sum(row["status"] == "complete" for row in result["rows"]) == 768
    assert sum(row["status"] == "failed" for row in result["rows"]) == 768
    assert result["numerical_training_calls"] == sum(
        x["calls"] for x in measured["gradient_call_ledger"] if x["cluster_id"] == plan["stream_id"]
    )

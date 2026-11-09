"""REQ-REPORT-8947 / SCENARIO-REPORT-8947-CONTRACT: released-label selection."""

from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from carnot.memory import gradient_replay_8947 as replay
from carnot.reporting import gradient_replay_methods_8947 as m
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def params() -> dict:
    """Use the serving feature map so controls exercise the actual objective."""
    return m.signed.head(11, "signed", [1 / 3, 1 / 3])


def example(group: str, label: str = "support", value: float = 1.0) -> dict:
    """Keep labels explicit so feature collisions cannot hide target changes."""
    return dict(group_id=group, label=label, counts=[value] * 8, split="train")


def test_gradient_direction_permutations_and_ties() -> None:
    """REQ-REPORT-8947: equal features with opposite labels need different choices."""
    p = params()
    current = example("current")
    a, b = example("a", "refute"), example("b")
    candidates = replay.scan(p, current, [a, b])
    assert replay.select(candidates) == "b"
    assert candidates[1]["cosine"] < 0 < candidates[2]["cosine"]
    assert (
        replay.select(replay.scan(p, current, [dict(a, label="support"), dict(b, label="refute")]))
        == "a"
    )
    assert replay.select(replay.scan(p, current, [b, dict(a, label="support")])) == "a"
    assert replay.select(replay.scan(p, current, [a, dict(b, label="refute")])) == "a"
    assert replay.select(replay.scan(p, current, [])) == "current"
    assert replay.cosine(np.zeros((3, 9)), np.ones((3, 9))) == 0
    assert replay.cosine(np.ones((3, 9)), np.zeros((3, 9))) == 0
    assert replay.cosine(np.ones((3, 9)), -np.ones((3, 9))) == pytest.approx(-1)
    with pytest.raises(ValueError, match="candidate_boundary"):
        replay.scan(p, current, [current])
    with pytest.raises(ValueError, match="candidate_boundary"):
        replay.scan(p, current, [a, a])
    with pytest.raises(ValueError, match="candidate_boundary"):
        replay.scan(p, current, [dict(a, split="guard")])
    with pytest.raises(ValueError):
        replay.scan(p, current, [dict(a, counts=[float("nan")] * 8)])
    assert all(r["passed"] for r in replay.selector_controls(p))


def test_delayed_release_equal_scans_and_restart(tmp_path: Path) -> None:
    """REQ-REPORT-8947: pending labels cannot enter scans; corruption stays in TRAIN."""
    p = params()
    stores = {a: replay.Store(tmp_path / (a + ".json"), p, a, m.anchors()) for a in m.ARMS}
    batch = []
    for i in range(48):
        group = f"gradient-control/{i}"
        value = m.old.world(m.old.graphs()["development"][i % 8], group, m.old.ERRORS[i % 6], i)
        batch.append((group, value))
        for store in stores.values():
            with patch.object(replay.g.symbolic, "decide", side_effect=AssertionError("future")):
                store.predict(group, value, i + 1, i + 9)
    for store in stores.values():
        before = deepcopy(store.state)
        with pytest.raises(ValueError):
            store.release(batch[:8], 15)
        assert store.state == before
        for offset in range(0, 48, 8):
            store.release(batch[offset : offset + 8], offset + 16)
        state = store.state
        train = [r for r in state["released"] if r["split"] == "train"]
        count = len(train) if state["arm"] != "frozen_verified" else 0
        assert state["training_calls"] == 2 * count == len(state["gradient_ledger"])
        assert state["evaluated_gradients"] == count + len(state["candidate_gradient_rows"])
        assert len(state["buffer"]) == 32
        assert replay.Store(store.path, {}, state["arm"], m.anchors()).state == state
        assert all(
            r["label"] == r["true_label"] for r in state["released"] if r["split"] == "guard"
        )
        for row in state["selection_rows"]:
            assert row["released_id"] not in row["buffer_ids"]
            assert row["scan_count"] == len(row["buffer_ids"]) + 1
        assert all(r["split"] == "train" for r in state["candidate_gradient_rows"])
    states = [stores[a].state for a in m.ARMS[1:]]
    assert len({s["evaluated_gradients"] for s in states}) == 1
    assert len({s["training_calls"] for s in states}) == 1
    assert any(r["label"] != r["true_label"] for r in states[-1]["released"])
    assert all(s["templates"] == stores[m.ARMS[0]].state["templates"] for s in states)
    assert stores[m.ARMS[0]].state["committed"] == p


def test_protocol_fresh_sealed_eleven_bounds() -> None:
    """SCENARIO-REPORT-8947-CONTRACT: only development worlds can be opened."""
    p = m.protocol()
    assert len(p["pilot_manifest"]) == 8 and len(p["frozen_evaluation_manifest"]) == 24
    assert p["arms"] == list(m.ARMS) and p["feature_schema"] == "signed-local-v1"
    assert p["training_window"] == 32 and p["learning_rate"] == 0.1
    assert not p["evaluation_opened"]
    e = p["evaluation_plan"]
    assert len(e["bounds"]) == e["family_intervals"] == 11
    assert e["bootstrap_resamples"] == 10000 and e["alpha_per_bound"] == 0.01 / 11
    assert {b["against"] for b in e["bounds"] if b["split"] == "online"} == set(m.ARMS) - {
        "gradient_replay"
    }
    assert all(s["delays"] == [8] * 48 for s in p["frozen_evaluation_manifest"])
    assert len({s["seed"] for s in p["pilot_manifest"] + p["frozen_evaluation_manifest"]}) == 32


@pytest.fixture(scope="module")
def measured(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict]:
    """Run paired development once; evaluation seeds remain recipes only."""
    raw = tmp_path_factory.mktemp("gradient-replay")
    return raw, m.measure(raw, {str(s): params() for s in m.old.SEEDS})


def test_mechanism_and_recovery(measured: tuple[Path, dict], tmp_path: Path) -> None:
    """REQ-REPORT-8947: readiness measures exercise, never development benefit."""
    raw, result = measured
    assert len(result["rows"]) == m.COUNT == 18432
    assert all(r["passed"] for r in result["control_rows"])
    assert all(r["value"] == r["numerator"] / r["denominator"] for r in result["rows"])
    assert len({(r["unit_id"], r["arm"], r["metric"]) for r in result["rows"]}) == m.COUNT
    assert result["selection_disagreement"]["numerator"] > 0
    assert result["selection_disagreement"]["nonzero_gradients"] > 0
    assert m.exercised(result)
    assert not m.exercised(
        dict(result, selection_disagreement=dict(numerator=0, nonzero_gradients=0))
    )
    assert result["numerical_training_calls"] > result["applied_gradient_count"] > 0
    assert m.recover(raw, "interrupted") == result
    plan = m.protocol()["pilot_manifest"][0]
    continuous = m.run_stream(tmp_path / "continuous", plan, params(), restart=False)
    resumed = json.loads((raw / (plan["stream_id"] + ".json")).read_bytes())
    assert continuous["states"] == resumed["states"]
    assert all(
        s["prediction_cursor"] == 48 and s["cursor"] == 40 and len(s["pending"]) == 8
        for s in continuous["states"].values()
    )
    assert not list(raw.glob("*evaluation*"))


def test_external_authentication_and_private_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-8947: retained inputs are byte evidence, never retired score gates."""
    from carnot.reporting import gradient_replay_runner_8947 as r
    from carnot.reporting.v746_retained_evidence import reference

    source = tmp_path / "retained.json"
    monkeypatch.setattr(r, "PINS", {"retained": "sha256:absent"})
    raw = tmp_path / "raw"
    assert r.preconditions(raw, tmp_path)["diagnostics"][0]["check"] == "missing_file"
    source.write_text("bad json")
    assert not r.preconditions(raw, tmp_path)["external_ready"]
    atomic_json(source, {})
    monkeypatch.setattr(r, "PINS", {"retained": reference(source)["sha256"]})
    assert r.preconditions(raw, tmp_path)["diagnostics"][0]["check"] == "missing_field"
    assert (
        r.preconditions(raw, tmp_path)["diagnostics"][0]["artifact_field"]
        == "source_artifact_hashes"
    )
    operand = tmp_path / "heads.json"
    atomic_json(operand, {"11": params()})
    atomic_json(
        source,
        dict(
            source_artifact_hashes=[],
            code_config_hashes=[],
            raw_shard_hashes=[reference(operand)],
            retired_ready_score=0,
        ),
    )
    monkeypatch.setattr(r, "PINS", {"retained": reference(source)["sha256"]})
    assert r.preconditions(raw, tmp_path)["external_ready"]
    operand.write_text("changed")
    assert r.preconditions(raw, tmp_path)["diagnostics"][0]["check"] == "rejected_producer"
    operand.unlink()
    assert r.preconditions(raw, tmp_path)["diagnostics"][0]["check"] == "missing_file"
    output = tmp_path / (m.NAME + ".json")
    assert r.main(["--private-output", str(output), "--sources", str(tmp_path / "absent")]) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked" and value["gradient_methods_ready_score"] == 0
    assert r.main(["--cold-replay", str(output)]) == 0
    value["rows"][0]["value"] = 0.5
    value.pop("reproducibility_checksum")
    value["reproducibility_checksum"] = canonical_hash(value)
    atomic_json(output, value)
    assert r.main(["--cold-replay", str(output)]) == 1
    with pytest.raises(SystemExit):
        r.main(["--date", "20261108"])
    with pytest.raises(SystemExit):
        r.main(["--private-output", str(m.ROOT / "results" / (m.NAME + ".json"))])


@pytest.mark.parametrize("mode", ["null", "resources", "measurement", "validation", "terminal"])
def test_terminal_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, measured: tuple[Path, dict], mode: str
) -> None:
    """SCENARIO-REPORT-8947-CONTRACT: false owned controls cannot grant readiness."""
    from types import SimpleNamespace
    from carnot.reporting import gradient_replay_runner_8947 as r
    from carnot.reporting.v746_retained_evidence import reference

    _, result = measured
    heads = tmp_path / "heads.json"
    atomic_json(heads, {str(s): params() for s in m.old.SEEDS})
    monkeypatch.setattr(
        r,
        "preconditions",
        lambda *args: dict(
            external_ready=True,
            resources=dict(passed=mode != "resources"),
            references=[reference(heads)],
            diagnostics=[],
        ),
    )
    validation = dict(
        coverage=dict(
            complete=mode != "validation",
            percent=100,
            files=[dict(path=n, missing=[], excluded=[]) for n in m.OWNED],
        ),
        union=dict(passed=True),
        receipts=[dict(passed=True)],
        raw_references=[],
    )
    log = tmp_path / "terminal.log"
    log.write_text("private terminal control")
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

    def measurement(raw: Path, values: dict) -> dict:
        if mode == "measurement":
            raise ValueError("owned_failure")
        atomic_json(raw / "heads.json", values)
        return deepcopy(result)

    monkeypatch.setattr(m, "measure", measurement)
    output = tmp_path / (m.NAME + ".json")
    assert r.main(["--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == ("null" if mode == "null" else "disqualified")
    assert r.replay(output)["passed"]
    for key, changed in [
        ("gradient_methods_ready_score", 1 - value["gradient_methods_ready_score"]),
        ("numerical_training_calls", -1),
        ("protocol_hash", "changed"),
        ("completed_count", -1),
    ]:
        bad = deepcopy(value)
        bad[key] = changed
        bad.pop("reproducibility_checksum")
        bad["reproducibility_checksum"] = canonical_hash(bad)
        atomic_json(output, bad)
        with pytest.raises(ValueError):
            r.replay(output)
    if mode == "null":
        bad = deepcopy(value)
        measurement_path = Path(value["raw_directory"]) / "measurement.json"
        changed = json.loads(measurement_path.read_bytes())
        changed["candidate_gradient_rows"][0]["cosine"] += 0.1
        atomic_json(measurement_path, changed)
        bad["candidate_gradient_rows"] = changed["candidate_gradient_rows"]
        bad["raw_shard_hashes"] = [
            reference(measurement_path) if ref["path"] == str(measurement_path) else ref
            for ref in bad["raw_shard_hashes"]
        ]
        bad.pop("reproducibility_checksum")
        bad["reproducibility_checksum"] = canonical_hash(bad)
        atomic_json(output, bad)
        with pytest.raises(ValueError, match="semantic_drift"):
            r.replay(output)


def test_actual_child_cli(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8947-CONTRACT: real child replay works outside the checkout."""
    import os
    import subprocess
    import sys

    env = dict(os.environ, JAX_PLATFORMS="cpu")
    env.pop("PYTHONPATH", None)
    output = tmp_path / (m.NAME + ".json")
    for args, expected in [
        (["--private-output", str(output), "--sources", str(tmp_path / "missing")], 0),
        (["--cold-replay", str(output)], 0),
        (["--cold-replay", str(tmp_path / "missing.json")], 1),
        (["--cold-replay", str(output)], 1),
    ]:
        if expected == 1 and str(output) in args:
            bad = json.loads(output.read_bytes())
            bad["rows"][0]["value"] = 0.3
            bad.pop("reproducibility_checksum")
            bad["reproducibility_checksum"] = canonical_hash(bad)
            atomic_json(output, bad)
        print(f"[exp8947-test] phase=child_before completed=0 pending=1 args={args}", flush=True)
        child = subprocess.run(
            [sys.executable, "-u", str(m.ROOT / m.CLI), *args],
            cwd=tmp_path,
            env=env,
            text=True,
            capture_output=True,
            timeout=120,
        )
        print(child.stdout + child.stderr, end="", flush=True)
        print("[exp8947-test] phase=child_after completed=1 pending=0", flush=True)
        assert child.returncode == expected, child.stdout + child.stderr


def test_w1_gradients_exact_update_and_active_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-8947: scan at W1 and apply the chosen gradient without a third update."""
    from carnot.reporting import gradient_replay_runner_8947 as r
    from carnot.reporting.v746_retained_evidence import reference

    p = params()
    store = replay.Store(tmp_path / "store.json", p, "gradient_replay", m.anchors())
    state = deepcopy(store.state)
    row = example("current")
    prior = example("prior", "refute")
    state["buffer"] = [prior]
    state["visible_counts"] = row["counts"]
    w1, _ = m.signed.update(p, row["counts"], row["label"])
    expected, _ = m.signed.update(w1, prior["counts"], prior["label"])
    store.train_release(state, row)
    assert state["candidate"] == expected
    assert all(c["at_weights"] == canonical_hash(w1) for c in state["candidate_gradient_rows"])
    assert state["training_calls"] == 2 and state["evaluated_gradients"] == 3
    before = deepcopy(store.state)
    with patch.object(replay.g, "atomic_json", side_effect=OSError("disk")):
        with pytest.raises(OSError):
            store.commit(before)
    assert store.state == before
    name = "experiment_8944_v765_branch_contract"
    path = tmp_path / (name + ".json")
    for score, check in [
        (0, "observed_zero"),
        (False, "observed_zero"),
        (True, "rejected_producer"),
    ]:
        atomic_json(path, dict(contract_ready_score=score))
        monkeypatch.setattr(r, "PINS", {name: reference(path)["sha256"]})
        assert r.preconditions(tmp_path / "raw", tmp_path)["diagnostics"][0]["check"] == check
    assert r.adapter().m is m
    with patch.object(r, "stream", return_value={"passed": True}) as mocked:
        assert r.adapter().stream("collect_once", [], tmp_path)["passed"]
        assert mocked.call_args.kwargs["deadline"] == 180
        assert r.adapter().stream("other", [], tmp_path)["passed"]


def test_failed_and_partial_recovery(
    measured: tuple[Path, dict], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-8947: failed attempts and completed seals survive interruption."""
    import shutil

    raw, _ = measured
    plan = m.protocol()["pilot_manifest"][0]
    shutil.copytree(
        raw / "states" / plan["stream_id"], tmp_path / "partial" / "states" / plan["stream_id"]
    )
    result = m.recover(tmp_path / "partial", "interrupted")
    assert 0 < sum(r["status"] == "complete" for r in result["rows"]) < m.COUNT
    assert result["numerical_training_calls"] > 0

    def fail(*args: object, **kwargs: object) -> dict:
        raise ValueError("owned_stream_failure")

    monkeypatch.setattr(m.reuse, "run_stream", fail)
    with pytest.raises(ValueError, match="owned_stream_failure"):
        m.measure(tmp_path / "failed", {str(s): params() for s in m.old.SEEDS})
    result = m.recover(tmp_path / "failed", "owned_stream_failure")
    assert sum(r["status"] == "failed" for r in result["rows"]) == 2304
    assert sum(r["status"] == "censored" for r in result["rows"]) == m.COUNT - 2304

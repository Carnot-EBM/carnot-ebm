"""REQ-REPORT-8933 / SCENARIO-REPORT-8933-CONTRACT: causal replay controls."""

from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from carnot.memory import compatible_replay_8933 as replay
from carnot.reporting import compatible_replay_methods_8933 as m
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def params() -> dict:
    """Keep tests on the same normalized signed feature schema as serving."""
    return m.signed.head(11, "signed", [1 / 3, 1 / 3])


def test_distance_label_blind_ties_and_collisions() -> None:
    """REQ-REPORT-8933: feature collisions use unit id, never labels or input order."""
    p = params()
    a = dict(group_id="a", counts=[1.0] * 8, label="support")
    b = dict(group_id="b", counts=[1.0] * 8, label="refute")
    far = dict(group_id="c", counts=[2.0] * 8, label="unknown")
    assert replay.nearest(p, a["counts"], [far, b, a]) == "a"
    assert replay.nearest(p, a["counts"], [dict(b, label="support"), a]) == "a"
    assert replay.nearest(p, far["counts"], [a, far]) == "c"
    with pytest.raises(ValueError, match="duplicate"):
        replay.nearest(p, a["counts"], [a, a])
    for bad in (float("nan"), float("inf")):
        with pytest.raises(ValueError):
            replay.nearest(p, [bad] * 8, [a])
    assert replay.selector_controls(p) and all(r["passed"] for r in replay.selector_controls(p))


def test_prediction_release_restart_and_budgets(tmp_path: Path) -> None:
    """REQ-REPORT-8933: predict before labels, then charge precisely two TRAIN gradients."""
    p = params()
    stores = {a: replay.Store(tmp_path / (a + ".json"), p, a, m.anchors()) for a in m.ARMS}
    batch = []
    for i in range(48):
        group = f"control/{i}"
        value = m.old.world(m.old.graphs()["development"][i % 8], group, m.old.ERRORS[i % 6], i)
        batch.append((group, value))
        for store in stores.values():
            with patch.object(replay.g.symbolic, "decide", side_effect=AssertionError("future")):
                store.predict(group, value, i + 1, i + 9)
        if i == 0:
            before = deepcopy(stores[m.ARMS[1]].state)
            with pytest.raises(ValueError):
                stores[m.ARMS[1]].release(batch, 8)
            assert stores[m.ARMS[1]].state == before
    changed_labels = False
    for store in stores.values():
        for offset in range(0, 48, 8):
            store.release(batch[offset : offset + 8], offset + 16)
        state = store.state
        train = [r for r in state["released"] if r["split"] == "train"]
        expected = 0 if state["arm"] == "frozen_verified" else 2 * len(train)
        assert state["training_calls"] == expected == len(state["gradient_ledger"])
        assert [r["group_id"] for r in state["buffer"]] == [r["group_id"] for r in train[-32:]]
        assert len(state["buffer"]) == 32
        assert all(r["split"] == "train" for r in state["buffer"])
        if expected:
            assert state["selection_rows"][0]["empty_buffer"]
            assert all(r["optimizer_steps"] == 1 for r in state["gradient_ledger"])
        resumed = replay.Store(store.path, {}, state["arm"], m.anchors())
        assert resumed.state == state
        changed_labels |= any(r["label"] != r["true_label"] for r in state["released"])
        assert all(
            r["label"] == r["true_label"] for r in state["released"] if r["split"] == "guard"
        )
        with pytest.raises(ValueError):
            store.release(batch[:8], 100)
    assert changed_labels
    baseline = stores["frozen_verified"].state
    assert baseline["committed"] == p
    assert all(s.state["templates"] == baseline["templates"] for s in stores.values())
    store = stores["compatible_replay"]
    before = deepcopy(store.state)
    with patch.object(replay.g, "atomic_json", side_effect=OSError("disk")):
        with pytest.raises(OSError):
            store.commit(deepcopy(before))
    assert store.state == before
    broken = deepcopy(before)
    broken["candidate"]["weights"][0][0][0] = float("nan")
    atomic_json(store.path, broken)
    with pytest.raises(ValueError):
        replay.Store(store.path, {}, "compatible_replay", m.anchors())
    with pytest.raises(ValueError):
        replay.Store(tmp_path / "bad.json", p, "other", m.anchors())


def test_protocol_and_sealed_evaluation() -> None:
    """SCENARIO-REPORT-8933-CONTRACT: freeze ten bounds before opening evaluation."""
    p = m.protocol()
    assert len(p["pilot_manifest"]) == 8 and len(p["frozen_evaluation_manifest"]) == 24
    assert not p["evaluation_opened"] and p["training_window"] == 32
    assert p["evaluation_plan"]["bootstrap_resamples"] == 10000
    assert p["evaluation_plan"]["alpha_per_bound"] == 0.001
    assert len(p["evaluation_plan"]["bounds"]) == 10
    assert p["arms"] == list(m.ARMS)
    assert len({s["seed"] for s in p["pilot_manifest"] + p["frozen_evaluation_manifest"]}) == 32
    assert all(s["delays"] == [8] * 48 for s in p["frozen_evaluation_manifest"])


@pytest.fixture(scope="module")
def measured(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict]:
    """REQ-REPORT-8933: run real tiny-head arithmetic on development worlds only."""
    raw = tmp_path_factory.mktemp("compatible-replay")
    result = m.measure(raw, {str(s): params() for s in m.old.SEEDS})
    return raw, result


def test_measured_controls_and_rows(measured: tuple[Path, dict]) -> None:
    """SCENARIO-REPORT-8933-CONTRACT: methods readiness is independent of effect size."""
    raw, result = measured
    assert len(result["rows"]) == 15360
    assert all(r["passed"] for r in result["control_rows"])
    assert result["numerical_training_calls"] == len(result["gradient_ledger"])
    assert all(r["value"] == r["numerator"] / r["denominator"] for r in result["rows"])
    assert len({(r["unit_id"], r["arm"], r["metric"]) for r in result["rows"]}) == 15360
    assert not list(raw.glob("*evaluation*"))
    assert m.recover(raw, "interrupted") == result
    assert len(m.census("missing")["rows"]) == 15360
    assert all(r["status"] == "censored" for r in m.census("missing")["rows"])


def test_external_gates_and_blocked_cli(tmp_path: Path) -> None:
    """REQ-REPORT-8933: absent, malformed and rejected producers terminate before training."""
    from carnot.reporting import compatible_replay_runner_8933 as r

    raw = tmp_path / "raw"
    assert r.preconditions(raw, tmp_path)["diagnostics"][0]["check"] == "missing_file"
    name, (_, field, _) = next(iter(r.PINS.items()))
    for value, expected in [
        ({}, "missing_field"),
        ({field: 0}, "observed_zero"),
        ({field: 1}, "rejected_producer"),
    ]:
        atomic_json(tmp_path / (name + ".json"), value)
        assert r.preconditions(raw, tmp_path)["diagnostics"][0]["check"] == expected
    (tmp_path / (name + ".json")).write_text("invalid")
    assert not r.preconditions(raw, tmp_path)["external_ready"]
    output = tmp_path / (m.NAME + ".json")
    assert r.main(["--sources", str(tmp_path), "--private-output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == "blocked" and value["replay_methods_ready_score"] == 0
    assert r.main(["--replay", str(output)]) == 0
    value["rows"][0]["value"] = 0.3
    atomic_json(output, value)
    assert r.main(["--replay", str(output)]) == 1
    with pytest.raises(SystemExit):
        r.main(["--date", "20261109"])
    with pytest.raises(SystemExit):
        r.main(["--private-output", str(m.ROOT / "results" / (m.NAME + ".json"))])


@pytest.mark.parametrize("mode", ["null", "resources", "measurement", "validation", "terminal"])
def test_terminal_paths_and_rehashed_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, measured: tuple[Path, dict], mode: str
) -> None:
    """SCENARIO-REPORT-8933-CONTRACT: owned failures disqualify, completed methods stay null."""
    from types import SimpleNamespace
    from carnot.reporting import compatible_replay_runner_8933 as r
    from carnot.reporting.v746_retained_evidence import reference

    _, result = measured
    fit = tmp_path / "fit.json"
    atomic_json(fit, dict(heads=dict(signed={"11": params()})))
    pre = dict(
        external_ready=True,
        resources=dict(passed=mode != "resources"),
        references=[reference(fit)],
        diagnostics=[],
    )
    monkeypatch.setattr(r, "preconditions", lambda *args: pre)
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

    def measurement(raw: Path, heads: dict) -> dict:
        if mode == "measurement":
            raise ValueError("owned_failure")
        atomic_json(raw / "heads.json", heads)
        return deepcopy(result)

    monkeypatch.setattr(m, "measure", measurement)
    output = tmp_path / (m.NAME + ".json")
    assert r.main(["--output", str(output)]) == 0
    value = json.loads(output.read_bytes())
    assert value["verdict_class"] == ("null" if mode == "null" else "disqualified")
    assert r.replay(output)["passed"]
    for key, changed in [
        ("replay_methods_ready_score", 1 - value["replay_methods_ready_score"]),
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


def test_successful_custody_and_adapter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-8933: original operand hashes are mandatory even after producer admission."""
    from carnot.reporting import compatible_replay_runner_8933 as r
    from carnot.reporting.v746_retained_evidence import reference

    source = tmp_path / "experiment_8919_fixture.json"
    operand = tmp_path / "operand"
    operand.write_text("original bytes")
    atomic_json(
        source,
        dict(
            experiment_id=8919,
            task_id="exp8919-fixture",
            milestone="2026.11.763",
            required_checks_passed=True,
            ready=1,
            source_artifact_hashes=[reference(operand)],
            code_config_hashes=[],
            raw_shard_hashes=[],
        ),
    )
    precedent = tmp_path / r.PRECEDENT
    precedent.write_text("original selector")
    monkeypatch.setattr(r, "PRECEDENT_PIN", reference(precedent)["sha256"])
    monkeypatch.setattr(
        r, "PINS", {source.stem: (reference(source)["sha256"], "ready", "exp8919-fixture")}
    )
    assert r.preconditions(tmp_path / "raw", tmp_path)["external_ready"]
    operand.write_text("changed")
    assert not r.preconditions(tmp_path / "raw", tmp_path)["external_ready"]
    operand.unlink()
    assert any(
        d["check"] == "missing_file"
        for d in r.preconditions(tmp_path / "raw", tmp_path)["diagnostics"]
    )
    assert r.adapter().m is m


def test_actual_private_cli(tmp_path: Path) -> None:
    """SCENARIO-REPORT-8933-CONTRACT: actual child and cold replay run outside checkout."""
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
        print(f"[exp8933-test] before_cli arguments={arguments}", flush=True)
        child = subprocess.run(
            [sys.executable, "-u", str(m.ROOT / m.CLI), *arguments],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        print(child.stdout + child.stderr, end="", flush=True)
        print(f"[exp8933-test] after_cli exit={child.returncode}", flush=True)
        assert child.returncode == expected, child.stdout + child.stderr


def test_partial_recovery_and_restart_equivalence(
    measured: tuple[Path, dict], tmp_path: Path
) -> None:
    """REQ-REPORT-8933: restart preserves future choices and interrupted seals remain counted."""
    import shutil

    raw, _ = measured
    plan = m.protocol()["pilot_manifest"][0]
    state_dir = tmp_path / "partial" / "states" / plan["stream_id"]
    shutil.copytree(raw / "states" / plan["stream_id"], state_dir)
    result = m.recover(tmp_path / "partial", "interrupted")
    assert 0 < sum(r["status"] == "complete" for r in result["rows"]) < 15360
    assert result["numerical_training_calls"] == len(result["gradient_ledger"]) > 0
    uninterrupted = m.run_stream(tmp_path / "uninterrupted", plan, params(), restart=False)
    restarted = json.loads((raw / (plan["stream_id"] + ".json")).read_bytes())
    assert uninterrupted["states"] == restarted["states"]


def test_failed_attempt_census(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-8933: failed owned attempts remain distinguishable from unopened streams."""

    def fail(*args: object) -> dict:
        raise ValueError("owned_stream_failure")

    monkeypatch.setattr(m, "run_stream", fail)
    with pytest.raises(ValueError, match="owned_stream_failure"):
        m.measure(tmp_path, {str(s): params() for s in m.old.SEEDS})
    recovered = m.recover(tmp_path, "owned_stream_failure")
    assert sum(r["status"] == "failed" for r in recovered["rows"]) == 1920
    assert sum(r["status"] == "censored" for r in recovered["rows"]) == 13440


def test_retained_original_store_custody(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-REPORT-8933: authenticate the retained historical store instead of current bytes."""
    from carnot.reporting import compatible_replay_runner_8933 as r
    from carnot.reporting.v746_retained_evidence import reference

    pin = "sha256:379c78d6c15c75ca6e4410bf1b08a8c65b9276f27c562cd23cd6b8faf1dd6033"
    original = (
        m.ROOT
        / "results/raw/experiment_8919_v763_interpolated_update_methods/original_sources"
        / pin.split(":")[1]
    )
    historical_bytes = original.read_bytes()
    monkeypatch.setattr(m, "ROOT", tmp_path)
    retained = (
        tmp_path
        / "results/raw/experiment_8919_v763_interpolated_update_methods/original_sources"
        / pin.split(":")[1]
    )
    retained.parent.mkdir(parents=True)
    retained.write_bytes(historical_bytes)
    source = tmp_path / "experiment_8919_fixture.json"
    atomic_json(
        source,
        dict(
            experiment_id=8919,
            task_id="exp8919-fixture",
            milestone="2026.11.763",
            required_checks_passed=True,
            ready=1,
            source_artifact_hashes=[
                dict(path=str(tmp_path / "python/carnot/memory/guarded_memory_8905.py"), sha256=pin)
            ],
            code_config_hashes=[],
            raw_shard_hashes=[],
        ),
    )
    precedent = tmp_path / r.PRECEDENT
    precedent.write_text("selector")
    monkeypatch.setattr(r, "PRECEDENT_PIN", reference(precedent)["sha256"])
    monkeypatch.setattr(
        r, "PINS", {source.stem: (reference(source)["sha256"], "ready", "exp8919-fixture")}
    )
    assert r.preconditions(tmp_path / "raw", tmp_path)["external_ready"]
    with patch.object(r, "stream", return_value={"passed": True}) as mocked:
        assert r.adapter().stream("collect_once", [], tmp_path)["passed"]
        assert mocked.call_args.kwargs["deadline"] == 180
        assert r.adapter().stream("other", [], tmp_path)["passed"]

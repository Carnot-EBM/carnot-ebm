"""REQ-REPORT-8145 / REQ-VERIFY-8145: natural costs and honest absent categories."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import runpy

import numpy as np
import pytest

from carnot import experiment_8145_v704_natural_service_cost as e
from carnot.reporting import natural_service_execution_8145 as runner


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    """Authenticate original natural evidence once, outside publication storage."""
    return e.inputs(e.ROOT, tmp_path_factory.mktemp("8145-input"))


@pytest.fixture(scope="module")
def native(data):
    """E2E-003 crosses the exact previously qualified extension."""
    return e.prior.old.host.load_binding(data)[0]


@pytest.fixture
def small():
    """A private panel exercises each lifecycle without asserting natural speed."""
    return dict(e.CONFIG, batches=[1, 4], repetitions=2, warmups=1, bootstraps=10000)


def test_inputs_and_gate(data, tmp_path):
    """REQ-REPORT-8145: deferred candidates cannot become rejected admissions."""
    assert data["ready"] and data["categories"] == {"accepted": 60, "rejected": 0}
    assert data["updates"] and data["head"]["optimizer_step"] == 4
    assert len(data["public"]) == 192 and any(r["values"] is None for r in data["public"])
    blocked = e.inputs(tmp_path / "absent", tmp_path / "blocked")
    assert not blocked["ready"]
    assert blocked["checks"][0]["artifact_field"] == "resource_exists"
    assert blocked["checks"][0]["observed"] is False
    assert e.CONFIG["measurement_ceiling_s"] == 2400
    assert len(e.prior.schedule(e.CONFIG)) == 900


@pytest.mark.parametrize("condition", e.prior.CONDITIONS)
def test_loaded_natural_transactions(data, native, tmp_path, condition):
    """SCENARIO-VERIFY-8145: offset, typed actions, full cost and E2E-004 restore."""
    slot = dict(batch=4, repetition=0, condition=condition, unit_id="private")
    rows = [e.transaction(data, native, slot, arm, tmp_path / arm) for arm in e.prior.ARMS]
    assert e.prior.parity(rows)
    for row in rows:
        expected = [
            e.engine.scalar_probability(data["head"], data["geometry"], v) for v in row["values"]
        ]
        np.testing.assert_allclose(row["probabilities"], expected, atol=1e-10, rtol=0)
        assert row["full_latency_ns"] >= sum(row["components"].values()) > 0
        state = json.loads(Path(row["state_path"]).read_text())["state"]
        restored = native.RustRadial8105.restore(
            native.RustRadial8105(json.dumps(state)).checkpoint()
        )
        assert json.loads(restored.state_json()) == state
    assert (
        e.recovery_fixture(data, native, tmp_path / "recovery")["verdict_class"]
        == "circular_positive"
    )


def test_update_admission_and_absent_rejections(data, native, tmp_path):
    """REQ-VERIFY-8145: original accepted event is replayed and durably charged."""
    rows = [
        e.update_transaction(data, native, data["updates"][0], arm, tmp_path / arm)
        for arm in e.prior.ARMS
    ]
    assert e.prior.parity(rows)
    assert all(r["natural_disposition"] == "accepted" for r in rows)
    assert all(r["components"]["candidate_fit_ns"] > 0 for r in rows)
    assert all(r["steps"] == data["updates"][0]["event"]["steps"] for r in rows)


def test_measure_reduction_censoring_and_replay(data, native, small, tmp_path):
    """REQ-REPORT-8145: completed timing, absent category and mutations stay distinct."""
    raw = tmp_path / "raw"
    work = e.measure(data, native, raw, small)
    value = e.build(data, work, raw, [dict(passed=True)], "20261005", 1, False)
    assert value["natural_service_ready_score"] == 1
    assert value["natural_update_cost_ready_score"] == 0
    assert value["rejected_update_cost"] is None
    assert value["MODEL_SPECS"] == [] and not value["call_ledger"]
    candidate = tmp_path / "candidate.json"
    e.atomic_json(candidate, value)
    assert e.replay(candidate)
    changed = deepcopy(value)
    changed["natural_update_cost_ready_score"] = 1
    changed["reproducibility_checksum"] = e.checksum(changed)
    e.atomic_json(candidate, changed)
    assert not e.replay(candidate)
    e.atomic_json(candidate, value)
    Path(work["pairs"][0]["arms"][0]["state_path"]).write_text("{}")
    assert not e.replay(candidate)
    assert not e.replay(tmp_path / "missing.json")
    assert (
        e.build(data, work, raw, [dict(passed=False)], "20261005", 1, False)["verdict_class"]
        == "disqualified"
    )
    censored = e.measure(data, native, tmp_path / "censored", dict(small, measurement_ceiling_s=0))
    assert all(p["status"] == "censored" for p in censored["pairs"])
    assert e.reduce_rows(censored)["censored_count"] == 24


def test_script_outside_checkout(tmp_path):
    """SCENARIO-REPORT-8145: direct failure and missing cold input need no PYTHONPATH."""
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env.pop("PYTHONPATH", None)
    command = [str(e.ROOT / ".venv/bin/python"), "-u", str(e.ROOT / e.CLI)]
    for args, expected in [
        (["--date", "invalid"], 2),
        (["--cold-replay", str(tmp_path / "absent")], 1),
    ]:
        result = subprocess.run(
            command + args, cwd=tmp_path, env=env, capture_output=True, timeout=60
        )
        assert result.returncode == expected


def test_update_and_input_corruption(data, native, tmp_path):
    """SCENARIO-VERIFY-8145: reject changed sealed training and admission operands."""
    changed = deepcopy(data["updates"][0])
    changed["before"]["candidates"]["error_center"]["head"]["weights"][0] += 1
    with pytest.raises(ValueError, match="candidate_fit_parity"):
        e.update_transaction(data, native, changed, "native_batch", tmp_path / "fit")
    changed = deepcopy(data["updates"][0])
    changed["event"]["steps"]["error_center"] = 0
    with pytest.raises(ValueError, match="admission_parity"):
        e.update_transaction(data, native, changed, "python_batch", tmp_path / "admit")
    root = tmp_path / "bad"
    e.atomic_json(root / e.UPSTREAM, {})
    assert not e.inputs(root, tmp_path / "bad_raw")["ready"]


def test_cli_private_success_block_replay_and_mutation(tmp_path):
    """SCENARIO-REPORT-8145: real script-path publication works outside the checkout."""
    env = dict(os.environ, PYTHONUNBUFFERED="1", JAX_PLATFORMS="cpu")
    env.pop("PYTHONPATH", None)
    command = [str(e.ROOT / ".venv/bin/python"), "-u", str(e.ROOT / e.CLI)]
    output = tmp_path / (e.NAME + ".json")
    result = subprocess.run(
        command + ["--private-small", "--output", str(output)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout.decode() + result.stderr.decode()
    value = json.loads(output.read_text())
    assert (
        value["verdict_class"] == "circular_positive" and value["natural_service_ready_score"] == 0
    )
    assert (
        subprocess.run(
            command + ["--cold-replay", str(output)],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            timeout=60,
        ).returncode
        == 0
    )
    value["rows"][0]["numerator"] += 1
    e.atomic_json(output, value)
    assert (
        subprocess.run(
            command + ["--cold-replay", str(output)],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            timeout=60,
        ).returncode
        == 1
    )
    blocked = tmp_path / "blocked" / (e.NAME + ".json")
    result = subprocess.run(
        command + ["--private-small", "--root", str(tmp_path / "absent"), "--output", str(blocked)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout.decode() + result.stderr.decode()
    assert json.loads(blocked.read_text())["verdict_class"] == "blocked"


def test_orchestration_owned_branches(data, native, small, tmp_path, monkeypatch):
    """REQ-REPORT-8145: owned failures and preserved primaries cannot grant readiness."""
    config = dict(small, batches=[1], repetitions=1, warmups=0)
    monkeypatch.setattr(e, "CONFIG", config)
    output = tmp_path / (e.NAME + ".json")
    commands = runner.validation_plan(tmp_path)
    assert any("--strict" in c.argv for c in commands)
    assert runner.run_owned(
        [
            runner.CommandSpec(
                "normal", (str(e.ROOT / ".venv/bin/python"), "-c", "print('done')"), "private", 5
            )
        ],
        tmp_path,
    )[0]["passed"]
    monkeypatch.setattr(
        runner,
        "run_owned",
        lambda commands, private: [
            dict(name=c.name, passed=True, normal_exit=True) for c in commands
        ],
    )
    assert runner.main(["--output", str(output)]) == 0
    assert runner.main(["--output", str(output)]) == 0
    assert list(output.parent.glob("raw/**/preserved_primary.json"))
    assert runner.main(["--cold-replay", str(output)]) == 0
    assert runner.main(["--cold-replay", str(tmp_path / "missing")]) == 1
    with pytest.raises(SystemExit) as raised:
        runner.main(["--private-small"])
    assert raised.value.code == 2
    monkeypatch.setattr(
        runner,
        "run_owned",
        lambda commands, private: [
            dict(name=c.name, passed=False, normal_exit=True) for c in commands
        ],
    )
    assert runner.main(["--output", str(tmp_path / "failed" / output.name)]) == 1
    monkeypatch.setattr(runner, "main", lambda: 0)
    with pytest.raises(SystemExit) as raised:
        runpy.run_path(str(e.ROOT / e.CLI), run_name="__main__")
    assert raised.value.code == 0


def test_rehashed_decision_and_custody_mutations(data, native, small, tmp_path):
    """SCENARIO-REPORT-8145: even rehashed decisions need independent replay."""
    raw = tmp_path / "raw"
    work = e.measure(data, native, raw, dict(small, batches=[1], repetitions=1, warmups=0))
    value = e.build(data, work, raw, [dict(passed=True)], "20261005", 1, False)
    assert (
        e.build(data, work, raw, [dict(passed=True)], "20261005", 1, True)["verdict_class"]
        == "circular_positive"
    )
    path = tmp_path / "candidate.json"
    for key in ["checksum", "source", "code"]:
        changed = deepcopy(value)
        if key == "checksum":
            changed["reproducibility_checksum"] = "invalid"
        else:
            if key == "source":
                changed["source_artifact_hashes"][0]["sha256"] = "invalid"
            else:
                changed["code_config_hashes"][e.OWNED[0]] = "invalid"
            changed["reproducibility_checksum"] = e.checksum(changed)
        e.atomic_json(path, changed)
        assert not e.replay(path)
    for kind in ["state", "probability"]:
        changed_work = deepcopy(work)
        for arm in changed_work["pairs"][0]["arms"]:
            if kind == "state":
                arm["durable_state"]["version"] = 123
            else:
                arm["probabilities"][0] += 0.001
        e.atomic_json(raw / "primitive_rows.json", changed_work)
        changed = deepcopy(value)
        changed["primitive_rows"] = e.reference(raw / "primitive_rows.json")
        changed["raw_shard_hashes"] = [
            e.reference(Path(r["path"])) for r in value["raw_shard_hashes"]
        ]
        changed["reproducibility_checksum"] = e.checksum(changed)
        e.atomic_json(path, changed)
        assert not e.replay(path)

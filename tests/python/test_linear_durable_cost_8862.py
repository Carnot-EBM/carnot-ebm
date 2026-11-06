"""REQ-REPORT-8862: complete paired linear costs need genuine durable work."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import linear_durable_cost_8862 as m


def test_frozen_census():
    """SCENARIO-REPORT-8862-CONTRACT: sixty cells retain matched48-query blocks."""
    rows = m.roster()
    assert rows == m.roster() and len(rows) == 60
    assert len({r["unit_id"] for r in rows}) == 60
    for block in range(10):
        cells = [r for r in rows if r["block"] == block]
        assert {(r["occupancy"], r["arm"]) for r in cells} == {
            (n, a) for n in (0, 32) for a in m.ARMS
        }
        assert all(r["denominator"] == 48 and r["seed"] == cells[0]["seed"] for r in cells)
    assert len(m.workload(0)) == 48


@pytest.fixture(scope="module")
def observed(tmp_path_factory):
    """REQ-REPORT-8862: real subprocesses prepare32 templates and restart at24."""
    raw = tmp_path_factory.mktemp("linear-durable")
    pre = m.preflight(raw / "pre")
    assert not pre["failures"]
    spec = dict(m.roster()[0], occupancy=32, arm="linear_combined")
    prep = m.execute(raw / "prep", spec, pre["heads"][str(spec["seed"])], preparation=True)
    row = m.execute(raw / "window", spec, pre["heads"][str(spec["seed"])], prepared=prep)
    return raw, pre, spec, prep, row


def test_observed_durability(observed):
    """REQ-REPORT-8862: equal durable transactions include updates, tail and barriers."""
    _, _, _, prep, row = observed
    assert m.parity(prep)["passed"] and m.parity(row)["passed"]
    assert row["query_count"] == row["attempted_updates"] == row["applied_updates"] == 48
    assert row["fsync_counts"] == dict(file=98, directory=98)
    assert row["template_hits"] == 24 and row["template_additions"] == 0
    assert row["parameter_movement"] > 0
    for key, bad in [("query_count", 47), ("complete_ns", -1), ("status", "failed")]:
        assert not m.parity(dict(row, **{key: bad}))["passed"]
    bad = deepcopy(row)
    bad["segments"][1]["restart_input_hash"] = "wrong"
    assert not m.parity(bad)["passed"]


@pytest.mark.parametrize("arm", m.ARMS)
def test_empty_and_frozen(observed, tmp_path, arm):
    """REQ-REPORT-8862: frozen and calibration-only arms retain equal causal work."""
    _, pre, spec, _, _ = observed
    spec = dict(spec, occupancy=0, arm=arm)
    prep = m.execute(tmp_path / "prep", spec, pre["heads"][str(spec["seed"])], preparation=True)
    row = m.execute(tmp_path / "window", spec, pre["heads"][str(spec["seed"])], prepared=prep)
    assert m.parity(prep)["passed"] and m.parity(row)["passed"]
    assert row["applied_updates"] == (0 if arm == "linear_frozen" else 48)
    assert row["final_occupancy"] == (32 if arm == "linear_combined" else 0)
    assert (
        row["parameter_movement"] == 0 if arm == "linear_frozen" else row["parameter_movement"] > 0
    )


def test_failed_child(observed, tmp_path):
    """REQ-REPORT-8862: fixed deadlines retain failed children without invented data."""
    _, pre, spec, prep, _ = observed
    row = m.execute(tmp_path, spec, pre["heads"][str(spec["seed"])], prepared=prep, deadline=0.001)
    assert row["status"] == "failed" and not m.parity(row)["passed"]


def test_reductions():
    """REQ-REPORT-8862: paired intervals and preparation costs reduce only observations."""
    rows = [dict(r, status="completed", complete_ns=48000 + r["block"]) for r in m.roster()]
    reduced = m.reduce_costs(rows, [])
    assert len(reduced["paired_cost_intervals"]) == 6
    assert all(r["paired_blocks"] == 10 for r in reduced["paired_cost_intervals"])
    assert len(reduced["occupancy_rows"]) == 6
    assert len(reduced["preparation_amortization_rows"]) == 18
    assert m.reduce_costs([], [])["paired_cost_intervals"][0]["lower_ns"] is None
    rows[0]["status"] = "failed"
    assert any(r["missing_pairs"] for r in m.reduce_costs(rows, [])["paired_cost_intervals"])


def test_authority_failures(tmp_path, monkeypatch):
    """REQ-REPORT-8862: missing, zero and altered inputs carry explicit upstream gates."""
    path = tmp_path / "authority.json"
    monkeypatch.setattr(m, "SOURCES", [(str(path), "wrong", "linear_methods_ready_score")])
    assert m.preflight(tmp_path / "missing")["failures"][0]["observed"] == "missing_file"
    m.durable(path, dict(linear_methods_ready_score=0, required_checks_passed=True))
    monkeypatch.setattr(
        m, "SOURCES", [(str(path), m.sha256_file(path), "linear_methods_ready_score")]
    )
    assert m.preflight(tmp_path / "zero")["failures"][0]["observed"] == "observed_zero"
    m.durable(path, dict(required_checks_passed=True))
    monkeypatch.setattr(
        m, "SOURCES", [(str(path), m.sha256_file(path), "linear_methods_ready_score")]
    )
    assert m.preflight(tmp_path / "field")["failures"][0]["observed"] == "missing_field"
    m.durable(
        path,
        dict(
            linear_methods_ready_score=1,
            required_checks_passed=True,
            code_config_hashes=[dict(path=str(tmp_path / "absent"), sha256="wrong")],
            raw_shard_hashes=[],
            source_artifact_hashes=[],
        ),
    )
    monkeypatch.setattr(
        m, "SOURCES", [(str(path), m.sha256_file(path), "linear_methods_ready_score")]
    )
    assert m.preflight(tmp_path / "operand")["failures"][0]["observed"] == "missing_file"


@pytest.fixture
def sealed(tmp_path, monkeypatch, observed):
    """REQ-REPORT-8862: failed owned preparations retain all sixty scheduled units."""
    from carnot.reporting import linear_durable_runner_8862 as r

    _, pre, _, prep, _ = observed
    monkeypatch.setattr(m, "preflight", lambda raw: deepcopy(pre))
    monkeypatch.setattr(
        m, "execute", lambda raw, spec, head, **kw: dict(prep, **spec, status="failed")
    )
    work = r.run(tmp_path / "run", "20261106")
    validation = dict(passed=True, coverage=dict(complete=True), receipts=[])
    m.durable(Path(work["raw"]) / "validation.json", validation)
    return work, validation


def test_build_and_replay(sealed, tmp_path):
    """SCENARIO-REPORT-8862-CONTRACT: failed owned work disqualifies; tampering fails."""
    from carnot.reporting import linear_durable_runner_8862 as r

    work, validation = sealed
    value = r.build(work, validation)
    assert value["verdict_class"] == "disqualified" and value["whole_cost_ready_score"] == 0
    assert value["intended_count"] == value["censored_count"] == 60
    path = tmp_path / (m.NAME + ".json")
    m.durable(path, value)
    assert m.replay(path)["passed"]
    m.durable(path, dict(value, whole_cost_ready_score=1))
    assert not m.replay(path)["passed"]
    m.durable(path, value)
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    argv = [sys.executable, "-u", str(m.ROOT / m.CLI), "--replay", str(path)]
    child = subprocess.run(argv, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60)
    assert child.returncode == 0 and "replay_passed" in child.stdout
    m.durable(path, {})
    assert (
        subprocess.run(argv, cwd=tmp_path, env=env, capture_output=True, timeout=60).returncode == 1
    )
    assert not m.replay(tmp_path / "absent")["passed"]
    work["preconditions"]["failures"] = [dict(observed="missing_file")]
    assert r.build(work, validation)["verdict_class"] == "blocked"
    validation["passed"] = False
    assert r.build(work, validation)["verdict_class"] == "disqualified"


def test_full_schedule(observed, tmp_path, monkeypatch):
    """REQ-REPORT-8862: admitted pilot keeps all cells and readiness does not need benefit."""
    from carnot.reporting import linear_durable_runner_8862 as r

    _, pre, _, prep, row = observed
    monkeypatch.setattr(m, "preflight", lambda raw: pre)
    monkeypatch.setattr(
        m,
        "execute",
        lambda raw, spec, head, **kw: dict(prep if kw.get("preparation") else row, **spec),
    )
    monkeypatch.setattr(m, "parity", lambda row: dict(unit_id=row["unit_id"], passed=True))
    monkeypatch.setattr(m, "microbenchmark", lambda head: dict(numerical_training_calls=1))
    work = r.run(tmp_path, "20261106")
    assert len(work["window_rows"]) == 60 and len(work["preparation_cost_rows"]) == 20
    validation = dict(passed=True, coverage=dict(complete=True), receipts=[])
    m.durable(tmp_path / "validation.json", validation)
    value = r.build(work, validation)
    assert value["whole_cost_ready_score"] == 1 and value["verdict_class"] == "null"


def test_validation_dispatch(sealed, monkeypatch, tmp_path):
    """REQ-REPORT-8862: validation, publication and CLI use the same frozen producer."""
    from carnot.reporting import linear_durable_runner_8862 as r

    work, validation = sealed
    monkeypatch.setattr(r.base, "validate", lambda raw: validation)
    monkeypatch.setattr(r.base, "terminal", lambda *args: dict(passed=True))
    assert r.validate(Path(work["raw"])) == validation
    assert r.terminal(tmp_path / "candidate", Path(work["raw"]))["passed"]
    monkeypatch.setattr(r, "validate", lambda raw: validation)
    result = r.qualify(work, tmp_path / (m.NAME + ".json"))
    assert result["whole_cost_ready_score"] == 0
    monkeypatch.setattr(r, "run", lambda *args: work)
    monkeypatch.setattr(r, "qualify", lambda *args: result)
    assert r.main(["--private-output", str(tmp_path / "out"), "--raw-dir", str(tmp_path)]) == 0
    assert r.main(["--output", str(tmp_path / "out")]) == 0
    monkeypatch.setattr(m, "worker", lambda *args: None)
    assert r.main(["--worker", str(tmp_path / "request")]) == 0
    assert r.main(["--replay", str(tmp_path / "absent")]) == 1
    with pytest.raises(SystemExit):
        r.main(["--date", "20261105"])


def test_microbenchmark(observed):
    """REQ-REPORT-8862: numerical work has its own clock and cannot claim timing shares."""
    _, pre, spec, _, _ = observed
    row = m.microbenchmark(pre["heads"][str(spec["seed"])])
    assert row["duration_ns"] > 0 and row["numerical_training_calls"] == 48
    assert len(row["movement_rows"]) == 48


def test_rehashed_lifecycle(sealed, observed, tmp_path):
    """REQ-REPORT-8862: rehashed primitives cannot bypass parity or snapshot bytes."""
    from carnot.reporting import linear_durable_runner_8862 as r

    work, validation = sealed
    _, _, _, prep, row = observed
    raw = Path(work["raw"])
    work["preparation_cost_rows"] = [deepcopy(prep)]
    work["window_rows"][0] = deepcopy(row)
    path = tmp_path / "candidate.json"

    def seal():
        m.durable(raw / "measurement.json", work)
        m.durable(path, r.build(work, validation))

    seal()
    assert m.replay(path)["passed"]
    work["window_rows"][0]["query_count"] = 47
    seal()
    assert not m.replay(path)["passed"]
    work["window_rows"][0] = deepcopy(row)
    work["window_rows"][0]["segments"][0]["state_file"]["sha256"] = "wrong"
    seal()
    assert not m.replay(path)["passed"]
    work["window_rows"][0] = deepcopy(row)
    seal()
    Path(value_ref := r.build(work, validation)["raw_shard_hashes"][0]["path"]).write_text("{}")
    assert value_ref.endswith("measurement.json") and not m.replay(path)["passed"]


def test_external_census(tmp_path, monkeypatch, observed):
    """REQ-REPORT-8862: external absence blocks every scheduled window explicitly."""
    from carnot.reporting import linear_durable_runner_8862 as r

    pre = deepcopy(observed[1])
    pre["failures"] = [dict(observed="missing_file")]
    monkeypatch.setattr(m, "preflight", lambda raw: pre)
    work = r.run(tmp_path, "20261106")
    assert len(work["window_rows"]) == 60
    assert all(row["missing_reason"] == "external_missing_file" for row in work["window_rows"])


def test_bound_validation_receipts(sealed):
    """REQ-REPORT-8862: coverage and validation logs are exact retained operands."""
    from carnot.reporting import linear_durable_runner_8862 as r

    work, validation = sealed
    raw = Path(work["raw"])
    log = raw / "owned.log"
    log.write_text("unit validation receipt\n")
    validation["receipts"] = [dict(log_path=str(log), log_sha256=m.sha256_file(log))]
    validation["coverage"]["combined_reference"] = m.reference(log)
    m.durable(raw / "validation.json", validation)
    assert m.reference(log) in r.build(work, validation)["raw_shard_hashes"]

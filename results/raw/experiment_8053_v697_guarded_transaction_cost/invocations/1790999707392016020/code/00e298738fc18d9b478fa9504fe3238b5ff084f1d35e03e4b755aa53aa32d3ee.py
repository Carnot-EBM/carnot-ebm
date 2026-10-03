"""REQ-REPORT-8053: real binding, guard decisions and terminal bytes stay auditable."""

import copy
import json
from pathlib import Path
import runpy
import sys

import numpy as np
import pytest

from carnot import experiment_8053_v697_guarded_transaction_cost as e
from carnot.verify import guarded_transaction_8053 as m


@pytest.fixture
def data():
    """Use public synthetic geometry so private checks supply no natural credit."""
    from test_native_transaction_8040 import data as original

    d = original.__wrapped__()
    d["cases"] = m.fixtures(d["head"])
    return d


@pytest.fixture
def native(tmp_path, data):
    """SCENARIO-REPORT-8053-PARITY: call the recorded extension, never a Python stub."""
    return e.prior.load_library(data["library_reference"], tmp_path)[0]


def test_guard_and_transactions(tmp_path, data, native, monkeypatch):
    """SCENARIO-REPORT-8053-PARITY/COST: all alphas, storage and restarts agree."""
    controls = m.parity(data, native, tmp_path)
    assert controls["parity_passed"]
    assert {"accepted", "rejected", "reset"} <= {w["class"] for w in data["cases"]}
    monkeypatch.setitem(m.CONFIG, "warmups", 1)
    monkeypatch.setitem(m.CONFIG, "repetitions", 2)
    result = m.measure(data, native, tmp_path)
    assert result["complete_numerical_transaction_speedup"]
    assert result["total_useful_decisions_per_second"] == []
    assert all(r["transaction_ns"] >= sum(r["components"].values()) for r in result["rows"])
    assert all(r["restart_equal"] for r in result["rows"])
    assert m.scaling(data["head"], native)["natural_credit"] == 0
    for arm in ["python", "native"]:
        with pytest.raises(ValueError):
            m.calculate(data, dict(data["cases"][0], labels=[2]), native, arm)
    monkeypatch.setitem(m.CONFIG, "budget_s", -1)
    with pytest.raises(TimeoutError):
        m.measure(data, native, tmp_path)


def test_current_custody(tmp_path):
    """REQ-REPORT-8053: original bytes and every replay operand must qualify."""
    d, failed = e.load_inputs(e.ROOT, tmp_path)
    assert not failed and len(d["cases"]) == 3960
    assert all(w["natural"] for w in d["cases"])
    assert e.load_inputs(tmp_path, tmp_path / "missing")[1]


def test_cli_routes(tmp_path, data, monkeypatch):
    """SCENARIO-REPORT-8053-TERMINAL: run valid/null/blocked and tampered CLI exits."""
    monkeypatch.setattr(e, "validate", lambda *a: ([], {}, []))
    monkeypatch.setitem(m.CONFIG, "warmups", 0)
    monkeypatch.setitem(m.CONFIG, "repetitions", 1)
    source = tmp_path / "fixture.json"
    e.atomic_json(source, data)
    output = tmp_path / (e.NAME + ".json")
    assert (
        e.main(["--fixture-input", str(source), "--output", str(output), "--validation-worker"])
        == 0
    )
    assert e.main(["--cold-replay", str(output)]) == 0
    value = json.loads(output.read_text())
    value["complete_numerical_transaction_speedup"] = []
    e.atomic_json(output, value)
    assert e.main(["--cold-replay", str(output)]) == 1
    source.unlink()
    assert e.main(["--root", str(tmp_path), "--output", str(output)]) == 0
    assert json.loads(output.read_text())["verdict_class"] == "blocked"
    monkeypatch.setattr(sys, "argv", [e.SCRIPT, "--cold-replay", str(output)])
    with pytest.raises(SystemExit) as process_exit:
        runpy.run_path(str(e.ROOT / e.SCRIPT), run_name="__main__")
    assert process_exit.value.code == 0


def test_natural_parity_and_boundary_guards(tmp_path, native, data, monkeypatch):
    """SCENARIO-REPORT-8053-PARITY: every original candidate and action threshold agree."""
    current, failures = e.load_inputs(e.ROOT, tmp_path / "custody")
    assert not failures
    assert m.parity(current, native, tmp_path / "parity")["parity_passed"]
    groups = {(w["arm"], w["class"]) for w in current["cases"]}
    subset = dict(
        current,
        cases=[
            next(w for w in current["cases"] if (w["arm"], w["class"]) == group)
            for group in sorted(groups)
        ],
    )
    monkeypatch.setitem(m.CONFIG, "warmups", 0)
    monkeypatch.setitem(m.CONFIG, "repetitions", 1)
    assert (
        len(
            m.measure(subset, native, tmp_path / "representatives")[
                "total_useful_decisions_per_second"
            ]
        )
        == 4
    )
    w = copy.deepcopy(data["cases"][1])
    w["delta"] = [-10.0] * 110
    assert m.calculate(data, w, native, "native")["result"]["diagnostics"][0]["reasons"] == [
        "brier",
        "typed_cost",
        "new_false_accept",
    ]
    for boundary in (0.1, 0.5):
        for p in (boundary - 1e-12, boundary, boundary + 1e-12):
            h = copy.deepcopy(data["head"])
            h.update(
                parameters=[0.0] * 110,
                decay_scale=1.0,
                calibration=[float(np.log(p / (1 - p))), 0.0],
            )
            x = np.zeros((8, 110))
            y = np.array([0, 1] * 4)
            a = m.guard(
                h, np.zeros(110), np.zeros(110), np.zeros(110), x, y, "feedback_constrained", None
            )[0]
            b = m.guard(
                h, np.zeros(110), np.zeros(110), np.zeros(110), x, y, "feedback_constrained", native
            )[0]
            assert m.comparison(a, b)["passed"]
    assert not m.comparison(dict(a, alpha=None), b)["passed"]
    missing = copy.deepcopy(data)
    missing["sources"][0]["features"][0] += 1
    with pytest.raises(ValueError, match="public_feature_drift"):
        m.calculate(missing, w, native, "python")
    bad = copy.deepcopy(data["cases"][0])
    bad.update(updates=[0], labels=[2])
    with pytest.raises(ValueError, match="label_contract"):
        m.calculate(data, bad, native, "native")


def test_validation_and_acceptance_routes(tmp_path, data, monkeypatch):
    """SCENARIO-REPORT-8053-TERMINAL: owned failures cannot grant numerical readiness."""
    plan = e.validation_plan(tmp_path)
    assert any(c.name == "repository_health" and c.timeout_s <= 120 for c in plan)
    assert all(c.timeout_s <= 300 for c in plan)
    coverage = {p: dict(summary=dict(num_statements=1, missing_lines=0)) for p in e.OWNED}
    e.atomic_json(tmp_path / "coverage.json", dict(files=coverage))
    real_commands = e.run_commands
    monkeypatch.setattr(
        e,
        "run_commands",
        lambda *a, **k: [
            dict(scope="owned", passed=True),
            dict(scope="repository_health", passed=False),
        ],
    )
    receipts, counts, health = e.validate(tmp_path / "logs", tmp_path)
    assert receipts[0]["passed"] and counts and not health[0]["passed"]
    monkeypatch.setattr(e, "run_commands", real_commands)
    monkeypatch.setattr(e, "load_inputs", lambda *a: (data, []))
    monkeypatch.setattr(e, "validate", lambda *a: (receipts, counts, health))
    monkeypatch.setitem(m.CONFIG, "warmups", 0)
    monkeypatch.setitem(m.CONFIG, "repetitions", 1)
    output = tmp_path / "qualified" / (e.NAME + ".json")
    assert e.main(["--output", str(output)]) == 0
    good = json.loads(output.read_text())
    assert good["native_transaction_ready_score"] == 1
    assert e.replay(output)["passed"]
    for change in [
        dict(verdict_class="blocked"),
        dict(parity_rows=[]),
        dict(rows=[dict(good["rows"][0], transaction_ns=0)]),
    ]:
        mutated = dict(good, **change)
        e.atomic_json(output, mutated)
        with pytest.raises(ValueError):
            e.replay(output)
    e.atomic_json(output, good)
    rows_path = Path(good["raw_directory"]) / "transaction_rows.json"
    original_bytes = rows_path.read_bytes()
    for mode in ("checkpoint", "span"):
        rows = json.loads(original_bytes)["rows"]
        cp = Path(rows[0]["checkpoint"]["path"])
        checkpoint_bytes = cp.read_bytes()
        if mode == "checkpoint":
            payload = json.loads(checkpoint_bytes)
            payload["acceptance"]["alpha"] = None
            e.atomic_json(cp, payload)
            rows[0]["checkpoint"] = e.reference(cp)
        else:
            rows[0]["transaction_ns"] += 1
        e.atomic_json(rows_path, dict(rows=rows))
        changed = dict(good, **m.summarize(rows, good["config"]))
        changed.update(raw_shard_hashes=[], checkpoint_references=[])
        e.atomic_json(output, changed)
        with pytest.raises(
            ValueError, match="checkpoint_drift" if mode == "checkpoint" else "span_drift"
        ):
            e.replay(output)
        cp.write_bytes(checkpoint_bytes)
    rows_path.write_bytes(original_bytes)
    e.atomic_json(output, good)
    monkeypatch.setattr(e.prior, "regression", lambda *a: dict(passed=False, exit_code=139))
    assert e.main(["--output", str(tmp_path / "regression-failed" / output.name)]) == 0
    monkeypatch.setattr(e.prior, "regression", lambda *a: dict(passed=True, exit_code=0))
    monkeypatch.setattr(
        m, "measure", lambda *a: (_ for _ in ()).throw(TimeoutError("owned_budget"))
    )
    assert e.main(["--output", str(tmp_path / "budget-failed" / output.name)]) == 0


def test_operand_and_storage_failures(tmp_path, data, native, monkeypatch):
    """REQ-REPORT-8053: corrupt operands and failed writes retain exact rejection paths."""
    import sqlite3
    from types import SimpleNamespace

    missing = copy.deepcopy(data)
    missing["sources"][0]["features"][0] += 1
    with pytest.raises(ValueError, match="parity_public_feature_drift"):
        m.parity(missing, native, tmp_path / "bad-public")
    empty = dict(data["cases"][0], guards=[], updates=[], labels=[])
    assert m.calculate(data, empty, native, "native")["result"]["status"] == "waiting_guard"
    monkeypatch.setitem(m.CONFIG, "warmups", 0)
    monkeypatch.setitem(m.CONFIG, "repetitions", 1)
    absent = dict(data, cases=[w for w in data["cases"] if w["class"] != "reset"])
    assert any(
        r["transaction_class"] == "reset"
        for r in m.measure(absent, native, tmp_path / "absent")["rows"]
    )
    real_transaction = m.transaction
    for mode, expected in [
        ("restart", "pair_restart_contract"),
        ("coefficients", "pair_coefficient_contract"),
    ]:

        def broken(*args, **kwargs):
            row = real_transaction(*args, **kwargs)
            if row["arm"] == "native":
                if mode == "restart":
                    row["restart_equal"] = False
                else:
                    row["effective"][0] += 1
            return row

        with monkeypatch.context() as patch:
            patch.setattr(m, "transaction", broken)
            with pytest.raises(ValueError, match=expected):
                m.measure(data, native, tmp_path / mode)
    connection = sqlite3.connect

    class CorruptedRead(sqlite3.Connection):
        def execute(self, sql, *args, **kwargs):
            if sql.startswith("SELECT payload"):
                return SimpleNamespace(fetchone=lambda: (b"changed journal",))
            return super().execute(sql, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(
            m.sqlite3, "connect", lambda *a, **k: connection(*a, **k, factory=CorruptedRead)
        )
        with pytest.raises(ValueError, match="journal_restart_drift"):
            real_transaction(
                data, data["cases"][0], native, "python", tmp_path / "corrupt" / "head.json"
            )
    trajectory = tmp_path / "trajectory"
    seed = trajectory / "seed-101"
    seed.mkdir(parents=True)
    db = connection(seed / "ledger.sqlite")
    db.execute("CREATE TABLE events(seq INTEGER,kind TEXT,identity TEXT,payload TEXT)")
    db.execute(
        "INSERT INTO events VALUES(1,?,?,?)",
        (
            "acceptance",
            "bad",
            json.dumps(dict(arm="unconstrained", before_coefficients=[999.0] * 110)),
        ),
    )
    db.commit()
    db.close()
    with pytest.raises(ValueError, match="original_state_chain"):
        e.workloads(trajectory, data)
    for identity, name, ready in [
        (
            8051,
            "experiment_8051_v697_feedback_constrained_learning",
            "learning_trajectory_ready_score",
        ),
        (8040, e.prior.NAME, "native_transaction_ready_score"),
    ]:
        p = tmp_path / "inputs" / "results" / (name + ".json")
        side = tmp_path / f"{identity}-side.json"
        report = tmp_path / f"{identity}-report.json"
        e.atomic_json(
            p,
            dict(
                experiment_id=identity,
                task_id=f"exp{identity}-fixture",
                flagged_adversarial=False,
                **{ready: 1},
                terminal_validation_sidecar_path=str(side),
                loaded_library_receipt=dict(path="/missing", sha256="sha256:absent"),
            ),
        )
        binding = dict(
            primary_path=str(p), primary_sha256=e.sha256_file(p), sidecar_path=str(report)
        )
        e.atomic_json(side, dict(publication=binding))
        e.atomic_json(report, dict(**binding, report=dict(passed=True)))
    assert (
        e.load_inputs(tmp_path / "inputs", tmp_path / "custody")[1][0]["artifact_field"]
        == "trajectory_contract"
    )
    report = tmp_path / "8051-report.json"
    doc = json.loads(report.read_text())
    doc["report"]["passed"] = False
    e.atomic_json(report, doc)
    assert any(
        r["artifact_field"] == "report.passed"
        for r in e.load_inputs(tmp_path / "inputs", tmp_path / "custody2")[1]
    )
    with monkeypatch.context() as patch:
        patch.setattr(e, "ROOT", tmp_path / "absent-tools")
        assert any(
            r["artifact_field"] == "resource_exists"
            for r in e.load_inputs(tmp_path, tmp_path / "tools")[1]
        )


def test_private_process_cli(tmp_path, data):
    """SCENARIO-REPORT-8053-TERMINAL: real script processes exit normally on every route."""
    fixture = tmp_path / "fixture.json"
    e.atomic_json(fixture, data)
    output = tmp_path / "valid" / (e.NAME + ".json")
    output.parent.mkdir(parents=True)
    py = str(e.ROOT / ".venv/bin/python")
    cli = (py, "-u", str(e.ROOT / e.SCRIPT))
    commands = [
        e.CommandSpec(
            "valid_null",
            (*cli, "--fixture-input", str(fixture), "--output", str(output), "--validation-worker"),
            "private",
            60,
        ),
        e.CommandSpec("valid_cold", (*cli, "--cold-replay", str(output)), "private", 60),
    ]
    receipts = e.run_commands(e.ROOT, commands, log_dir=tmp_path / "process-logs", heartbeat_s=10)
    assert all(r["exit_code"] == 0 and r["passed"] for r in receipts)
    assert json.loads(output.read_text())["native_transaction_ready_score"] == 0
    v = json.loads(output.read_text())
    v["rows"] = []
    e.atomic_json(output, v)
    row = e.run_commands(
        e.ROOT,
        [e.CommandSpec("tampered", (*cli, "--cold-replay", str(output)), "private", 60)],
        log_dir=tmp_path / "tamper-logs",
        heartbeat_s=10,
    )[0]
    assert row["exit_code"] == 1
    blocked = tmp_path / "blocked" / (e.NAME + ".json")
    blocked.parent.mkdir(parents=True)
    row = e.run_commands(
        e.ROOT,
        [
            e.CommandSpec(
                "blocked",
                (
                    *cli,
                    "--root",
                    str(tmp_path / "absent"),
                    "--output",
                    str(blocked),
                    "--validation-worker",
                ),
                "private",
                60,
            )
        ],
        log_dir=tmp_path / "blocked-logs",
        heartbeat_s=10,
    )[0]
    assert row["exit_code"] == 0 and json.loads(blocked.read_text())["verdict_class"] == "blocked"

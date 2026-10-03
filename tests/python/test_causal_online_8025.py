"""REQ-SELF-8025, REQ-REPORT-8025: causal sparse updates and durable CLI evidence."""

import copy
import json
from pathlib import Path
import sqlite3

import numpy as np
import pytest

from carnot import experiment_8025_v695_causal_online_updates as e
from carnot.verify import causal_online_8025 as m
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def fixture(n=64):
    """Artificial labels test causality without giving natural benefit credit."""
    head = dict(
        parameters=[0.0] * 110,
        decay_scale=1.0,
        calibration=[0.0, 1.0],
        geometry=dict(
            scaler=dict(minimum=[0.0] * 9, maximum=[1.0] * 9), logit_center=0.0, logit_scale=1.0
        ),
        arm="conditioned_energy",
    )
    sources = [
        dict(
            family_id=str(i),
            source_cluster_id=str(i),
            slot=i,
            q=0.5,
            features=[i / max(n, 1)] * 8,
            public_eligible=True,
            exclusion_reason=None,
        )
        for i in range(n)
    ]
    labels = {str(i): i % 2 for i in range(n)}
    return dict(head=head, sources=sources, labels=labels, seeds=[101])


def test_sparse_dense_and_selection():
    """SCENARIO-SELF-8025: sparse calibrated descent equals its dense equation."""
    data = fixture()
    head = data["head"]
    head["parameters"] = np.linspace(-0.2, 0.3, 110).tolist()
    head["calibration"] = [0.2, 1.7]
    x = m.design(head, data["sources"][4])
    before = m.coefficients(head)
    p = m.probability(head, x)
    expected = before - 0.01 * ((p - 1) * 1.7 * x + 0.002 * before)
    info = m.update(head, x, 1)
    assert np.allclose(m.coefficients(head), expected, atol=1e-14)
    assert info["hot_update_ns"] > 0 and info["bytes_written"] <= 39 * 8
    block = [dict(family_id=str(i), actual_cost=float(i), brier=float(15 - i)) for i in range(16)]
    assert [r["family_id"] for r in m.select(block, "decision_loss", 101)] == [
        "15",
        "14",
        "13",
        "12",
    ]
    assert [r["family_id"] for r in m.select(block, "brier_loss", 101)] == ["0", "1", "2", "3"]
    assert [r["family_id"] for r in m.select(block, "periodic", 101)] == ["0", "4", "8", "12"]
    assert m.select(block, "uniform", 101) == m.select(block, "uniform", 101)


def test_causal_unknown_budget_reload(tmp_path):
    """SCENARIO-SELF-8025: issue precedes release; unknowns never update."""
    data = fixture()
    data["labels"]["3"] = None
    data["sources"][5]["public_eligible"] = False
    value = m.measure(data, tmp_path)
    assert len(value["issued_rows"]) == 64 * 5
    for arm in m.ARMS:
        assert value["per_arm_budget"][f"{arm}/101"]["updates"] == (
            0 if arm == "frozen_no_write" else 8
        )
    assert all(r["due_slot"] == r["origin_slot"] + 20 for r in value["released_feedback_rows"])
    assert not any(r["family_id"] == "3" for r in value["update_rows"])
    assert m.reduce(tmp_path) == value
    changed = copy.deepcopy(data)
    changed["labels"]["40"] = 1 - changed["labels"]["40"]
    other = m.measure(changed, tmp_path / "other")
    assert [
        (r["probability"], r["head_hash"]) for r in value["issued_rows"] if r["slot"] <= 60
    ] == [(r["probability"], r["head_hash"]) for r in other["issued_rows"] if r["slot"] <= 60]
    db = sqlite3.connect(tmp_path / "ledger.sqlite")
    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            "INSERT INTO events(kind,identity,payload) SELECT kind,identity,payload FROM events LIMIT 1"
        )
    db.close()


def test_cli_branches_and_terminal(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8025: direct CLI, cold reduction, publication and blockage."""
    data = fixture()
    src = tmp_path / "fixture.json"
    atomic_json(src, data)
    output = tmp_path / "out" / (e.NAME + ".json")
    monkeypatch.setattr(e, "terminal", lambda p: dict(passed=True))
    assert (
        e.main(["--fixture-input", str(src), "--validation-worker", "--output", str(output)]) == 0
    )
    value = json.loads(output.read_text())
    assert value["learning_measurement_ready_score"] == 0
    assert value["retained_labels_opened"] is False
    assert e.main(["--cold-replay", str(output)]) == 0
    blocked = tmp_path / "blocked" / (e.NAME + ".json")
    assert (
        e.main(
            ["--root", str(tmp_path / "absent"), "--validation-worker", "--output", str(blocked)]
        )
        == 0
    )
    assert json.loads(blocked.read_text())["verdict_class"] == "blocked"
    assert e.main(["--cold-replay", str(blocked)]) == 0
    value["rows"][0]["probability"] = 0.9
    atomic_json(output, value)
    with pytest.raises(ValueError, match="reduction_drift"):
        e.replay(output)


def source_fixture(root, bad=None):
    """Private immutable upstream artifacts exercise production admission paths."""
    data = fixture(256)

    def save(name, value):
        path = root / (name + ".json")
        atomic_json(path, value)
        return e.reference(path)

    h = dict(data["head"], seed=17, converged=bad != "head")
    pub = save("public", dict(rows=data["sources"][:8] if bad == "slots" else data["sources"]))
    target = save(
        "targets", dict(rows=[dict(family_id=k, eligible_y=y) for k, y in data["labels"].items()])
    )
    upstream = dict(
        task_id=e.fitted.TASK,
        energy_fit_ready_score=1,
        verdict_class="null",
        flagged_adversarial=False,
        head_checkpoints=[save("head", h)],
        calibration_checkpoint=save(
            "calibration",
            dict(maps={"conditioned_energy-17": dict(converged=True, parameters=[0.0, 1.0])}),
        ),
    )
    admission = dict(
        task_id=e.eligible.TASK,
        stream_targets_ready_score=1,
        verdict_class="null",
        flagged_adversarial=False,
        support_by_role=dict(stream=dict(passed=bad != "support")),
        public_manifests=dict(stream=pub),
        evaluator_manifests=dict(stream=target),
        exclusion_manifest=save("exclusions", {}),
        historical_failure_logs=[dict(path=pub["path"], hash=pub["sha256"])],
    )
    atomic_json(root / "results" / (e.fitted.NAME + ".json"), upstream)
    atomic_json(root / "results" / (e.eligible.NAME + ".json"), admission)
    return data


def test_admission_and_natural_cli(tmp_path, monkeypatch):
    """REQ-REPORT-8025: original slots and the calibrated head gate admission."""
    for bad in ("support", "head", "slots", None):
        root = tmp_path / str(bad)
        source_fixture(root, bad)
        data, failures = e.load_inputs(root, root / "raw")
        assert bool(failures) == (bad is not None)
    monkeypatch.setitem(m.CONFIG, "seeds", [101])
    output = tmp_path / "natural" / (e.NAME + ".json")
    assert e.main(["--root", str(root), "--validation-worker", "--output", str(output)]) == 0
    assert e.replay(output)["passed"]
    document = json.loads(output.read_text())
    assert len(document["released_feedback_rows"]) == 5 * 236
    assert e.terminal(output)["passed"]
    document["learning_measurement_ready_score"] = 1
    atomic_json(output, document)
    with pytest.raises(ValueError, match="unsafe_readiness"):
        e.replay(output)


def test_owned_validation_and_cli_entry(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8025: owned failures disqualify; broad health stays separate."""
    import runpy
    import sys

    commands = e.validation_plan(tmp_path)
    assert any(c.name == "repository_health" for c in commands)
    counts = {p: dict(num_statements=1, missing_lines=0) for p in e.OWNED}
    receipts = [dict(scope="owned", passed=True), dict(scope="repository_health", passed=False)]
    value = e.base([])
    value["acceptance_gate_results"]["trajectory"] = True
    e.apply_validation(value, receipts, counts)
    assert value["learning_measurement_ready_score"] == 1
    e.apply_validation(value, [dict(scope="owned", passed=False)], counts)
    assert (
        value["verdict_class"] == "disqualified" and value["learning_measurement_ready_score"] == 0
    )
    blocked = e.base([dict(passed=False)])
    e.apply_validation(blocked, [], {})
    assert blocked["verdict_class"] == "blocked"
    src = tmp_path / "fixture.json"
    atomic_json(src, fixture(24))
    output = tmp_path / "full" / (e.NAME + ".json")
    real_plan = e.validation_plan

    def plan(scratch):
        result = real_plan(scratch)
        atomic_json(
            scratch / "coverage.json", dict(files={p: dict(summary=s) for p, s in counts.items()})
        )
        return result

    monkeypatch.setattr(e, "validation_plan", plan)
    monkeypatch.setattr(e, "run_commands", lambda *a, **kw: receipts)
    monkeypatch.setattr(e, "terminal", lambda p: dict(passed=True))
    assert e.main(["--fixture-input", str(src), "--output", str(output)]) == 0
    monkeypatch.setattr(sys, "argv", ["exp8025", "--cold-replay", str(output)])
    with pytest.raises(SystemExit) as stopped:
        runpy.run_path(str(e.ROOT / e.OWNED[-1]), run_name="__main__")
    assert stopped.value.code == 0
    monkeypatch.setattr(e, "reader_receipt", lambda *a, **kw: dict(passed=False))
    with pytest.raises(ValueError, match="published_validation_failed"):
        e.main(
            [
                "--root",
                str(tmp_path / "absent"),
                "--validation-worker",
                "--output",
                str(tmp_path / "rejected" / (e.NAME + ".json")),
            ]
        )


@pytest.mark.parametrize(
    "kind,field,value,message",
    [
        ("issue", "probability", 0.1, "issued_state_drift"),
        ("release", "due_slot", 0, "release_order"),
        ("release", "actual_cost", 99, "feedback_metric"),
        ("update", "family_id", "future", "selection_or_future_access"),
        ("update", "after_coefficients", [], "update_equation"),
    ],
)
def test_mutations(tmp_path, kind, field, value, message):
    """SCENARIO-SELF-8025: altered evidence cannot survive independent reduction."""
    m.measure(fixture(40), tmp_path)
    db = sqlite3.connect(tmp_path / "ledger.sqlite")
    seq, payload = db.execute(
        "SELECT seq,payload FROM events WHERE kind=? LIMIT 1", (kind,)
    ).fetchone()
    row = json.loads(payload)
    row[field] = value
    with db:
        db.execute("UPDATE events SET payload=? WHERE seq=?", (json.dumps(row), seq))
    db.close()
    with pytest.raises(ValueError, match=message):
        m.reduce(tmp_path)


def test_terminal_update_missing_and_checkpoint(tmp_path):
    """SCENARIO-SELF-8025: a missing final update or wrong checkpoint invalidates budgets."""
    raw = tmp_path / "missing"
    m.measure(fixture(36), raw)
    db = sqlite3.connect(raw / "ledger.sqlite")
    seq = db.execute("SELECT max(seq) FROM events WHERE kind='update'").fetchone()[0]
    with db:
        db.execute("DELETE FROM events WHERE seq=?", (seq,))
    db.close()
    with pytest.raises(ValueError, match="unequal_budget"):
        m.reduce(raw)
    raw = tmp_path / "checkpoint"
    m.measure(fixture(40), raw)
    db = sqlite3.connect(raw / "ledger.sqlite")
    initial = json.loads(
        db.execute("SELECT payload FROM events WHERE kind='issue' LIMIT 1").fetchone()[0]
    )
    seq, payload = db.execute(
        "SELECT seq,payload FROM events WHERE kind='update' LIMIT 1"
    ).fetchone()
    row = json.loads(payload)
    row["checkpoint"] = initial["checkpoint"]
    with db:
        db.execute("UPDATE events SET payload=? WHERE seq=?", (json.dumps(row), seq))
    db.close()
    with pytest.raises(ValueError, match="checkpoint_drift"):
        m.reduce(raw)


def test_invalid_label_and_positive_transition(tmp_path):
    """REQ-SELF-8025: bad labels fail; real sparse writes can change later fixture decisions."""
    data = fixture(40)
    data["labels"]["0"] = 2
    with pytest.raises(ValueError, match="target_contract"):
        m.measure(data, tmp_path / "invalid")
    data = fixture(64)
    data["head"]["parameters"][0] = -2.2
    data["labels"] = {str(i): 1 for i in range(64)}
    result = m.measure(data, tmp_path / "positive")
    assert any(r["changed_action"] for r in result["later_cost_rows"])
    assert any(r["gain"] > 0 for r in result["later_cost_rows"] if r["eligibility"])

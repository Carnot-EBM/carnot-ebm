"""REQ-VERIFY-8461, REQ-REPORT-8461: independent numerical and process evidence."""

from copy import deepcopy
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from carnot.verify import objective_checkpoint_8461 as e
from carnot.verify import objective_derivatives_8461 as d
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.v709_execution import child


def test_derivatives_and_negative_controls():
    """SCENARIO-VERIFY-8461-DERIVATIVES: every constructed cell has independent operands."""
    rows = d.checks(e.legacy.head())
    assert len(rows) == 90
    assert all(r["passed"] for r in rows)
    assert any(r["saturated"] for r in rows)
    assert all(
        r["negative_controls"]["reverse_gradient_rejected"] for r in rows if not r["saturated"]
    )
    assert all(
        r["negative_controls"]["omit_temperature_rejected"]
        for r in rows
        if r["temperature"] != 1 and abs(r["logit"]) <= 4
    )
    assert all(r["directional_decrease"] for r in rows if not r["stationary"])
    assert max(r["absolute_error"] for r in rows) <= 1e-7


def test_update_and_contract():
    """SCENARIO-VERIFY-8461-DERIVATIVES: logical global decay remains exact, including bias."""
    h = e.legacy.head()
    x = np.asarray([0.4] * 9)
    expected = e.sparse.parameters(h) - 0.01 * d.dense(h, x, 1)
    changed, receipt = e.update(h, x, 1)
    np.testing.assert_allclose(e.sparse.parameters(changed), expected, atol=1e-12, rtol=0)
    assert receipt["coefficient_touches"] <= 37
    assert e.update(h, x, 0, 0)[0] == h
    for field, value in [("arm", "mlp"), ("temperature", 0), ("decay_scale", 0)]:
        with pytest.raises(ValueError):
            e.update(dict(h, **{field: value}), x, 1)
    with pytest.raises(ValueError):
        e.update(h, x, 2)
    with pytest.raises(ValueError):
        e.update(h, np.array([float("nan")] * 9), 1)


def test_store_rules_labels_duplicates_and_semantics(tmp_path):
    """SCENARIO-VERIFY-8461-RECOVERY: saved hashes cannot authorize new rules or future labels."""
    cfg = e.config(e.legacy.head(), 8, "uniform", shuffled=True)
    store = e.Store(tmp_path, cfg)
    store.initialize()
    with pytest.raises(ValueError, match="unsealed"):
        store.release(0, {})
    ack = store.seal(0, e.legacy.questions(0))
    sealed = store.read()
    assert sealed["head"] == cfg["head"]
    assert store.seal(0, e.legacy.questions(0)) == ack
    labels = e.legacy.feedback(sealed)
    with pytest.raises(ValueError, match="release_roster"):
        store.release(0, dict(labels, future_label=1))
    ack = store.release(0, labels)
    assert store.release(0, labels) == ack
    bad = dict(labels)
    bad[next(iter(bad))] ^= 1
    with pytest.raises(ValueError, match="duplicate_conflict"):
        store.release(0, bad)
    result = store.read()
    assert e.replay_state(result, cfg)
    for field in ["head", "cursor", "admission_rng", "update_rule_hash"]:
        tamper = deepcopy(result)
        if field == "head":
            tamper[field]["parameters"][0] += 0.1
        elif field == "admission_rng":
            tamper[field][1][0] ^= 1
        elif field == "cursor":
            tamper[field] += 20
        else:
            tamper[field] = "forged"
        assert not e.replay_state(tamper, cfg)
    store.commit(dict(result, update_rule_hash="wrong"))
    with pytest.raises(ValueError, match="update_rule"):
        store.read()
    with pytest.raises(ValueError, match="update_rule"):
        e.config(dict(cfg["head"], update_rule="bad"))
    assert not e.replay_state(dict(result, version=-1), cfg)
    assert not e.replay_state({}, cfg)
    with pytest.raises(ValueError, match="duplicate_feedback"):
        e.release(
            dict(sealed, applied_feedback_ids=[f"feedback:0:{sealed['pending'][0]['id']}"]),
            cfg,
            0,
            labels,
        )
    frozen = e.config(e.legacy.head(), frozen=True)
    frozen_state = e.drive(tmp_path / "frozen", frozen)
    assert frozen_state["head"] == frozen["head"]


@pytest.mark.parametrize("transaction", ["seal", "release"])
@pytest.mark.parametrize(
    "barrier", ["temporary_write", "file_fsync", "atomic_replace", "acknowledgment"]
)
def test_actual_exits_and_exact_resume(tmp_path, transaction, barrier):
    """SCENARIO-VERIFY-8461-RECOVERY: a real private child exits on each commit side."""
    cfg = e.config(e.legacy.head(), 8, "uniform", shuffled=True)
    atomic_json(tmp_path / "cfg.json", cfg)
    cli = (
        Path(__file__).resolve().parents[2]
        / "scripts/experiments/experiment_8461_v728_objective_checkpoint_qualification.py"
    )
    argv = [
        sys.executable,
        "-u",
        str(cli),
        "--worker",
        str(tmp_path / "store"),
        "--configuration",
        str(tmp_path / "cfg.json"),
    ]
    crash = f"{transaction}:0:{barrier}"
    receipt = child("crash", argv + ["--crash", crash], tmp_path / "logs", expected=73, deadline=30)
    assert receipt["passed"]
    checkpoint = e.Store(tmp_path / "store", cfg).read()
    assert checkpoint["version"] == (0 if transaction == "seal" else 1) + int(
        barrier in ("atomic_replace", "acknowledgment")
    )
    assert e.replay_state(checkpoint, cfg)
    assert child("resume", argv, tmp_path / "logs", deadline=30)["passed"]
    actual = json.loads((tmp_path / "store/final.json").read_bytes())
    baseline = e.drive(tmp_path / "baseline", cfg)
    assert actual == baseline
    assert canonical_hash(actual["issued"]) == canonical_hash(baseline["issued"])


def test_transition_controls_and_invalid_configuration():
    """SCENARIO-VERIFY-8461-RECOVERY: future and early labels fail before any parameter update."""
    controls = d.transition_controls(e.legacy.head())
    assert all(r["passed"] for r in controls)
    cfg = e.config(e.legacy.head())
    state = e.bind(e.legacy.initial(cfg), cfg)
    assert not e.replay_state(state, dict(cfg, update_rule_hash="wrong"))
    assert not e.replay_state(state, {})


def test_negative_control_failure_and_rule_isolation(monkeypatch):
    """SCENARIO-VERIFY-8461-DERIVATIVES: controls expose an updater that silently accepts future labels."""
    original = deepcopy(e.RULE)
    cfg = e.config(e.legacy.head())
    cfg["update_rule"]["version"] = "wrong"
    assert e.RULE == original
    monkeypatch.setattr(e, "release", lambda state, cfg, block, labels: state)
    assert not any(r["passed"] for r in d.transition_controls(e.legacy.head()))


def test_nonfinite_update_and_checkpoint_rejected(tmp_path):
    """SCENARIO-VERIFY-8461-RECOVERY: a valid outer checksum cannot make nonfinite weights usable."""
    head = e.legacy.head()
    head["parameters"][0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        e.update(head, np.asarray([0.5] * 9), 1)
    store = e.Store(tmp_path, e.config(e.legacy.head()))
    store.initialize()
    state = store.read()
    state["head"]["parameters"][0] = float("inf")
    store.commit(state)
    with pytest.raises(ValueError, match="finite"):
        store.read()

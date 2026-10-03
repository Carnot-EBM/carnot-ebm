"""REQ-SELF-7999, REQ-VERIFY-7999: independent reconstruction and gates."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from carnot.verify import learning_causal_audit_7999 as m
from carnot.verify import selective_feedback_7998 as producer
from carnot.reporting.typed_validation_7997 import fixture_data


@pytest.fixture
def bundle(tmp_path):
    data = fixture_data()
    head, sources = data["heads"]["spline"][0], data["public"]["stream"]
    for r in sources:
        r["q"] = 0.5
    labels = {r["family_id"]: r["y"] for r in data["targets"]["stream"]}
    acquisition = producer.acquire(head, sources, 101)
    trajectories = {}
    for arm in m.ARMS:
        schedule = [r for r in acquisition if r["arm"] == arm]
        directory = tmp_path / arm
        trajectories[arm + "-101"] = dict(
            trajectory=producer.stream(head, sources, schedule, labels.__getitem__, directory),
            state_directory=str(directory),
        )
    return dict(
        head=head,
        sources=sources,
        acquisition=acquisition,
        trajectories=trajectories,
        seeds=[101],
        stream_targets=labels,
    )


def test_reconstruct_producer_gradients_and_durable_predictions(bundle):
    """REQ-VERIFY-7999: saved coefficient changes agree with independently derived gradients."""
    for arm in m.ARMS:
        schedule = [r for r in bundle["acquisition"] if r["arm"] == arm]
        saved = bundle["trajectories"][arm + "-101"]
        got = m.reduce(bundle["head"], bundle["sources"], schedule, saved, bundle["stream_targets"])
        assert got["final_state"] == saved["trajectory"]["final_state"]
        assert all(r["gradient_error"] < 1e-10 for r in got["gradient_rows"])
        assert got["prediction_changed"] == (arm != "frozen_no_write")
    assert m.controls()["passed"]
    assert m.controls()["known_benefit"]["verdict_class"] == "circular_positive"
    assert (
        m.controls()["known_benefit"]["final_cost"] < m.controls()["known_benefit"]["initial_cost"]
    )


@pytest.mark.parametrize("mutation", ["future_label", "missing_update", "propensity", "group"])
def test_independent_reducer_rejects_four_mutations(bundle, mutation):
    """SCENARIO-VERIFY-7999-MUTATION: self-consistent summaries cannot hide primitive tampering."""
    saved = copy.deepcopy(bundle["trajectories"]["targeted_ipw-101"])
    schedule = copy.deepcopy([r for r in bundle["acquisition"] if r["arm"] == "targeted_ipw"])
    if mutation == "future_label":
        saved["trajectory"]["reveal_rows"][0]["origin_slot"] += 1
    elif mutation == "missing_update":
        saved["trajectory"]["update_rows"].pop()
    elif mutation == "propensity":
        schedule[0]["pi"] = 0.9
    else:
        saved["trajectory"]["issued_predictions"][0]["source_cluster_id"] = "changed"
    with pytest.raises(ValueError):
        m.reduce(bundle["head"], bundle["sources"], schedule, saved, bundle["stream_targets"])


def test_recovery_both_commit_sides_and_corruption(bundle, tmp_path):
    """REQ-VERIFY-7999: disk restoration preserves RNG, pending IDs, weights and later outputs."""
    schedule = [r for r in bundle["acquisition"] if r["arm"] == "targeted_ipw"]
    for where in ("before", "after"):
        directory = tmp_path / where
        with pytest.raises(m.Crash):
            m.execute(
                bundle["head"],
                bundle["sources"],
                schedule,
                bundle["stream_targets"],
                directory,
                where,
            )
        got = m.execute(
            bundle["head"], bundle["sources"], schedule, bundle["stream_targets"], directory
        )
        expected = bundle["trajectories"]["targeted_ipw-101"]["trajectory"]
        assert got["final_state"] == expected["final_state"]
        assert got["issued_predictions"] == expected["issued_predictions"]
    path = directory / "state.json"
    value = json.loads(path.read_text())
    value["checksum"] = "bad"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="checkpoint"):
        m.execute(bundle["head"], bundle["sources"], schedule, bundle["stream_targets"], directory)


def test_bootstrap_support_seed_averaging_and_gate():
    """REQ-SELF-7999: schedules never enlarge sources; degenerate draws cannot pass benefit."""
    rows = [
        dict(
            source_cluster_id=str(i),
            slot=i,
            y=i % 2,
            eligibility=True,
            arm=a,
            seed=s,
            cost=0.1 if a == "targeted_ipw" else 0.25,
            brier=0.1,
            false_accept=0,
        )
        for i in range(40, 256)
        for a in m.ARMS
        for s in (101, 102)
    ]
    result = m.compare(rows)
    assert result["support_by_class"]["independent"] == 216
    assert result["effective_blocks"] == 10
    assert result["benefit"]
    for r in rows:
        r["cost"] = 0.25
    assert not m.compare(rows)["benefit"]
    assert not m.compare(rows[:10])["benefit"]
    assert m.bootstrap(np.zeros(64), 20)["interval"] == [0.0, 0.0]
    assert m.bootstrap(np.array([]), 20)["interval"] == [None, None]
    assert m.holm({"a": 0.01, "b": 0.04, "c": 0.9}) == {"a": 0.03, "b": 0.08, "c": 0.9}


def test_unavailable_and_unknown_targets(bundle):
    """REQ-SELF-7999: unavailable slots and unknown labels remain excluded."""
    sources = copy.deepcopy(bundle["sources"])
    sources[0].update(q=None, features=None, status="censored")
    sources[1]["q"] = 0.001
    schedule = m.acquire(bundle["head"], sources, 101, "full_feedback")
    labels = dict(bundle["stream_targets"])
    labels[sources[1]["family_id"]] = None
    got = m.execute(bundle["head"], sources, schedule, labels)
    assert got["issued_predictions"][0]["probability"] is None
    assert len(got["update_rows"]) == 234
    assert not m.acquire(bundle["head"], [sources[0]], 101, "uniform_ipw")[0]["selected"]
    with pytest.raises(ValueError, match="public"):
        m.acquire(bundle["head"], [dict(sources[1], y=1)], 101, "targeted_ipw")
    text = Path(m.__file__).read_text()
    assert "import selective_feedback_7998" not in text


def test_saved_checkpoint_mutations(bundle):
    """REQ-VERIFY-7999: gradients and intermediate state hashes are owned checks."""
    saved = copy.deepcopy(bundle["trajectories"]["targeted_ipw-101"])
    schedule = [r for r in bundle["acquisition"] if r["arm"] == "targeted_ipw"]
    saved["trajectory"]["checkpoint_rows"][0]["checksum"] = "bad"
    with pytest.raises(ValueError, match="state_drift"):
        m.reduce(bundle["head"], bundle["sources"], schedule, saved, bundle["stream_targets"])
    saved = bundle["trajectories"]["targeted_ipw-101"]
    slot = saved["trajectory"]["update_rows"][0]["due_slot"]
    path = Path(saved["state_directory"]) / f"committed-{slot:04d}.json"
    original = json.loads(path.read_text())
    changed = copy.deepcopy(original)
    changed["checksum"] = "bad"
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="saved_checkpoint"):
        m.reduce(bundle["head"], bundle["sources"], schedule, saved, bundle["stream_targets"])
    changed = copy.deepcopy(original)
    changed["state"]["head"]["parameters"][108] += 0.1
    from carnot.reporting.current_work_receipt import canonical_hash

    changed["checksum"] = canonical_hash(changed["state"])
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="saved_gradient"):
        m.reduce(bundle["head"], bundle["sources"], schedule, saved, bundle["stream_targets"])

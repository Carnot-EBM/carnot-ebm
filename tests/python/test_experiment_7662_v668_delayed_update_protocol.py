"""REQ-REPORT-7662: delayed updates use only causal, durable evidence."""

from __future__ import annotations

import json

import pytest

from carnot.reporting import experiment_7662_delayed_protocol as delayed
from carnot.reporting.experiment_7662_delayed_protocol import (
    DelayedUpdateService,
    source_stratum,
)


def _service(tmp_path, arm="source"):
    return DelayedUpdateService(tmp_path / f"{arm}.json", arm=arm)


def _predict(service, prefix, count, *, origin=0, partition="update"):
    for index in range(count):
        service.predict(
            f"{prefix}{index}",
            origin + index,
            0.35,
            index % 8,
            partition,
        )


def test_source_strata_preserve_partial_scope():
    """SCENARIO-REPORT-7662-CAUSAL: unknown evidence has its own stratum."""
    assert source_stratum(
        {"checked_structural_propositions": 0, "unknown_claims": 3}
    ) != source_stratum({"checked_structural_propositions": 1, "unknown_claims": 0})
    assert all(
        0 <= source_stratum({"checked_structural_propositions": n, "unknown_claims": n}) < 8
        for n in range(12)
    )


@pytest.mark.parametrize("variant", range(8))
@pytest.mark.parametrize(
    "attack",
    (
        "premature",
        "duplicate",
        "dropped",
        "out_of_order",
        "crash_before_ack",
        "replay_after_ack",
        "exhausted_admission",
        "rollback",
    ),
)
def test_64_adversarial_sequences(tmp_path, attack, variant):
    """SCENARIO-REPORT-7662-CAUSAL/DURABLE: 64 independent failure paths."""
    service = _service(tmp_path)
    _predict(service, "u", 5, origin=variant * 20)
    _predict(service, "a", 5, origin=variant * 20 + 5, partition="admission")
    before = service.state_hash
    first = f"u{variant % 5}"
    if attack == "premature":
        with pytest.raises(ValueError, match="feedback_too_early"):
            service.release(first, 1, variant * 20 + 7)
    elif attack == "duplicate":
        with pytest.raises(ValueError, match="duplicate_prediction"):
            service.predict("u0", variant * 20 + 10, 0.4, 0, "update")
    elif attack == "dropped":
        service.mark_missing("u0", variant * 20 + 20)
        assert service.feedback_status("u0") == "missing"
        assert service.state["numerical"]["count"] == [0.0] * 8
        with pytest.raises(ValueError, match="unreleased_update"):
            service.propose([f"u{i}" for i in range(5)])
    elif attack == "out_of_order":
        with pytest.raises(ValueError, match="release_order"):
            service.release("u1", 1, variant * 20 + 20)
    elif attack == "crash_before_ack":
        service.simulate_crash_before_ack()
        assert _service(tmp_path).state_hash == before
    elif attack == "replay_after_ack":
        service.release("u0", variant % 2, variant * 20 + 20)
        acknowledged = service.state_hash
        with pytest.raises(ValueError, match="duplicate_release"):
            _service(tmp_path).release("u0", variant % 2, variant * 20 + 21)
        assert _service(tmp_path).state_hash == acknowledged
    elif attack == "exhausted_admission":
        for index in range(5):
            service.release(f"u{index}", index % 2, variant * 20 + 20 + index)
        service.propose([f"u{i}" for i in range(5)])
        with pytest.raises(ValueError, match="admission_not_released"):
            service.admit([f"a{i}" for i in range(5)])
    else:
        for index in range(5):
            service.release(f"u{index}", 1, variant * 20 + 20 + index)
        service.propose([f"u{i}" for i in range(5)])
        for index in range(5):
            service.release(f"a{index}", 0, variant * 20 + 25 + index)
        result = service.admit([f"a{i}" for i in range(5)])
        assert result["accepted"] is False
        assert result["state_hash_after"] == result["prior_hash"]
    if attack in {"premature", "duplicate", "out_of_order"}:
        assert _service(tmp_path).state_hash == before


def test_service_e2e_and_arm_isolation(tmp_path):
    """SCENARIO-REPORT-7662-ADMISSION/DURABLE: persist and reload each state."""
    source = _service(tmp_path, "source")
    scalar = _service(tmp_path, "scalar")
    for service in (source, scalar):
        _predict(service, "u", 5)
        _predict(service, "a", 5, origin=5, partition="admission")
    scalar_before = scalar.state_hash
    for index in range(5):
        source.release(f"u{index}", 1, 10 + index)
    proposal = source.propose([f"u{i}" for i in range(5)])
    assert proposal["candidate_hash"] != proposal["prior_hash"]
    for index in range(5):
        source.release(f"a{index}", 1, 15 + index)
    outcome = source.admit([f"a{i}" for i in range(5)])
    assert outcome["admission_count"] == 5
    assert _service(tmp_path, "source").state_hash == source.state_hash
    assert scalar.state_hash == scalar_before
    assert len(json.loads(source.state_path.read_text())["numerical"]["count"]) == 8
    assert source.state_bytes < 100_000
    with pytest.raises(ValueError, match="admission_used"):
        source.admit([f"a{i}" for i in range(5)])


@pytest.mark.parametrize(
    ("args", "error"),
    [
        (("x", 0, 0.3, 0, "update"), "duplicate_prediction"),
        (("y", -1, 0.3, 0, "update"), "origin_order"),
        (("y", 1, 0.3, 0, "evaluation"), "partition_invalid"),
        (("y", 1, 0.3, 8, "update"), "stratum_invalid"),
        (("y", 1, float("nan"), 0, "update"), "probability_invalid"),
    ],
)
def test_invalid_predictions_are_atomic(tmp_path, args, error):
    """SCENARIO-REPORT-7662-CAUSAL: invalid predictions do not change state."""
    service = _service(tmp_path)
    service.predict("x", 0, 0.3, 0, "update")
    before = service.state_hash
    with pytest.raises(ValueError, match=error):
        service.predict(*args)
    assert _service(tmp_path).state_hash == before


def test_invalid_release_and_proposal_guards(tmp_path):
    """SCENARIO-REPORT-7662-ADMISSION: bad labels and batches are inert."""
    service = _service(tmp_path)
    _predict(service, "u", 5)
    _predict(service, "a", 5, origin=5, partition="admission")
    with pytest.raises(ValueError, match="unknown_prediction"):
        service.release("absent", 1, 20)
    with pytest.raises(ValueError, match="binary_label_required"):
        service.release("u0", 2, 20)
    service.release("u0", 1, 20)
    with pytest.raises(ValueError, match="release_order"):
        service.release("u1", 1, 19)
    for index in range(1, 5):
        service.release(f"u{index}", 1, 20 + index)
    with pytest.raises(ValueError, match="five_distinct_updates_required"):
        service.propose(["u0"] * 5)
    proposal = service.propose([f"u{i}" for i in range(5)])
    with pytest.raises(ValueError, match="proposal_pending"):
        service.propose([f"u{i}" for i in range(5)])
    with pytest.raises(ValueError, match="five_distinct_admissions_required"):
        service.admit(["a0"] * 5)
    for index in range(5):
        service.release(f"a{index}", 1, 25 + index)
    service.admit([f"a{i}" for i in range(5)])
    with pytest.raises(ValueError, match="proposal_missing"):
        service.admit([f"u{i}" for i in range(5)])
    with pytest.raises(ValueError, match="update_used"):
        service.propose([f"u{i}" for i in range(5)])
    assert proposal["candidate_hash"] != proposal["prior_hash"]


def test_state_capacity_and_reload_guards(tmp_path, monkeypatch):
    """SCENARIO-REPORT-7662-DURABLE: storage failures cannot acknowledge."""
    with pytest.raises(ValueError, match="arm_invalid"):
        _service(tmp_path, "wrong")
    service = _service(tmp_path)
    with pytest.raises(ValueError, match="arm_state_mismatch"):
        DelayedUpdateService(service.state_path, arm="scalar")
    monkeypatch.setattr(delayed, "STATE_LIMIT_BYTES", 1)
    with pytest.raises(ValueError, match="state_bytes_exceeded"):
        service.predict("x", 0, 0.3, 0, "update")
    monkeypatch.setattr(delayed, "STATE_LIMIT_BYTES", 100_000)
    real_atomic = delayed.atomic_json

    def corrupt(path, value):
        real_atomic(path, {**value, "acknowledgments": -1})

    monkeypatch.setattr(delayed, "atomic_json", corrupt)
    with pytest.raises(OSError, match="durable_reload_mismatch"):
        service._persist()


def test_stale_candidate_guard(tmp_path):
    """SCENARIO-REPORT-7662-DURABLE: a stale proposal cannot be admitted."""
    service = _service(tmp_path)
    _predict(service, "u", 5)
    _predict(service, "a", 5, origin=5, partition="admission")
    for index in range(5):
        service.release(f"u{index}", 1, 10 + index)
    service.propose([f"u{i}" for i in range(5)])
    for index in range(5):
        service.release(f"a{index}", 1, 15 + index)
    service.state["numerical"]["count"][0] = 1
    with pytest.raises(ValueError, match="stale_proposal"):
        service.admit([f"a{i}" for i in range(5)])

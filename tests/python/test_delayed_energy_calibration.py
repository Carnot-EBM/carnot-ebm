"""Tests for the bounded delayed affine adapter.

Spec refs: REQ-REPORT-7397, SCENARIO-REPORT-7397-AUTHORITY,
SCENARIO-REPORT-7397-FROZEN, and SCENARIO-REPORT-7397-CONTROLS.
"""

from __future__ import annotations

from copy import deepcopy
import math

import pytest

from carnot.learning.delayed_energy_calibration import (
    DelayedEnergyCalibrator,
    FutureLabelAccessError,
    affine_log_loss,
    energy,
    fit_affine,
    sigmoid,
)


def _weights() -> dict[str, object]:
    return {
        "w1": [[1.0, 0.0], [0.0, 1.0], [0.5, -0.5], [-0.5, 0.5]],
        "b1": [0.0, 0.0, 0.0, 0.0],
        "w_out": [1.0, -1.0, 0.5, -0.5],
        "b_out": 0.1,
    }


def test_sigmoid_energy_and_affine_fixtures() -> None:
    """REQ-REPORT-7397: informative and constant energies remain finite."""

    assert sigmoid(1000.0) == 1.0
    assert sigmoid(-1000.0) >= 0.0
    energies = [energy(_weights(), [value, 1.0 - value]) for value in (0.0, 0.3, 0.8, 1.0)]
    labels = [1, 1, 0, 0]
    before = affine_log_loss(energies, labels, 1.0, 0.0)
    fitted = fit_affine(energies, labels, steps=200)
    after = affine_log_loss(energies, labels, fitted["a"], fitted["b"])
    assert after < before
    assert 0.25 <= fitted["a"] <= 4.0
    assert -8.0 <= fitted["b"] <= 8.0

    constant = fit_affine([0.0, 0.0, 0.0, 0.0], labels, steps=10)
    assert math.isfinite(constant["loss_curve"][-1]["loss"])


def test_prediction_authority_duplicate_restart_and_erasure() -> None:
    """SCENARIO-REPORT-7397-AUTHORITY: one visible label gives one update."""

    adapter = DelayedEnergyCalibrator(_weights(), a=1.0, b=0.0)
    frozen_hash = adapter.weights_hash
    prediction = adapter.record_prediction(
        "event-1",
        [0.8, 0.2],
        prediction_index=0,
        feedback_available_at=1,
        label_authority="sealed-training-label",
    )
    assert prediction["prediction_before_feedback"] is True
    assert "label" not in prediction
    with pytest.raises(FutureLabelAccessError):
        adapter.commit_feedback("event-1", 1, visible_at=0)

    committed = adapter.commit_feedback("event-1", 1, visible_at=1)
    assert committed["status"] == "committed"
    assert adapter.update_count == 1
    assert adapter.weights_hash == frozen_hash
    state_after = adapter.state_hash

    duplicate = adapter.commit_feedback("event-1", 1, visible_at=2)
    assert duplicate["status"] == "duplicate"
    assert adapter.update_count == 1
    assert adapter.state_hash == state_after

    restored = DelayedEnergyCalibrator.from_dict(adapter.to_dict())
    assert restored.to_dict() == adapter.to_dict()
    assert restored.state_hash == adapter.state_hash

    erased = restored.erase_feedback("event-1")
    assert erased["status"] == "erased"
    assert restored.update_count == 0
    assert restored.a == 1.0
    assert restored.b == 0.0
    assert restored.weights_hash == frozen_hash


def test_missing_unknown_and_nonfinite_feedback_do_not_update() -> None:
    """SCENARIO-REPORT-7397-CONTROLS: malformed feedback cannot change state."""

    adapter = DelayedEnergyCalibrator(_weights(), a=4.0, b=8.0)
    adapter.record_prediction(
        "event-2",
        [1.0, 0.0],
        prediction_index=2,
        feedback_available_at=3,
        label_authority="sealed-training-label",
    )
    before = adapter.state_hash
    missing = adapter.commit_feedback("event-2", None, visible_at=3)
    assert missing["status"] == "missing"
    assert adapter.state_hash == before
    assert adapter.commit_feedback("unknown", 1, visible_at=9)["status"] == "unknown_event"
    assert adapter.erase_feedback("unknown")["status"] == "unknown_event"

    with pytest.raises(ValueError, match="finite"):
        adapter.record_prediction(
            "bad",
            [float("nan"), 0.0],
            prediction_index=3,
            feedback_available_at=4,
            label_authority="sealed-training-label",
        )
    with pytest.raises(ValueError, match="binary"):
        adapter.commit_feedback("event-2", 2, visible_at=3)
    with pytest.raises(ValueError, match="finite"):
        DelayedEnergyCalibrator({**deepcopy(_weights()), "b_out": float("inf")})
    with pytest.raises(ValueError, match="length"):
        fit_affine([0.0], [0, 1])
    with pytest.raises(ValueError, match="both labels"):
        fit_affine([0.0, 1.0], [1, 1])


def test_projection_and_gradient_cap_are_recorded() -> None:
    """REQ-REPORT-7397: every affine commit is clipped and projected."""

    adapter = DelayedEnergyCalibrator(_weights(), a=0.25, b=-8.0)
    adapter.record_prediction(
        "event-3",
        [100.0, -100.0],
        prediction_index=0,
        feedback_available_at=1,
        label_authority="fixture",
    )
    row = adapter.commit_feedback("event-3", 1, visible_at=1)
    assert row["gradient_norm_after_clip"] <= 1.0 + 1e-12
    assert 0.25 <= adapter.a <= 4.0
    assert -8.0 <= adapter.b <= 8.0
    assert row["gibbs_weights_unchanged"] is True


def test_adapter_rejects_every_malformed_state_boundary() -> None:
    """SCENARIO-REPORT-7397-CONTROLS: malformed state fails closed."""

    with pytest.raises(ValueError, match="numeric JSON"):
        DelayedEnergyCalibrator({**_weights(), "b_out": True})
    with pytest.raises(ValueError, match="contain"):
        DelayedEnergyCalibrator({"w1": []})
    with pytest.raises(ValueError, match="shape"):
        DelayedEnergyCalibrator({**_weights(), "w1": [[1.0, 2.0]]})
    with pytest.raises(ValueError, match="length"):
        DelayedEnergyCalibrator({**_weights(), "b1": [0.0]})
    with pytest.raises(ValueError, match="finite"):
        sigmoid(float("nan"))
    with pytest.raises(ValueError, match="common length"):
        affine_log_loss([], [], 1.0, 0.0)
    with pytest.raises(ValueError, match="binary"):
        affine_log_loss([0.0], [2], 1.0, 0.0)
    with pytest.raises(ValueError, match="between zero"):
        fit_affine([0.0, 1.0], [0, 1], steps=501)
    with pytest.raises(ValueError, match="finite"):
        fit_affine([0.0, float("nan")], [0, 1])
    with pytest.raises(ValueError, match="finite"):
        DelayedEnergyCalibrator(_weights(), a=float("nan"))
    with pytest.raises(ValueError, match="projection"):
        DelayedEnergyCalibrator(_weights(), a=0.1)
    with pytest.raises(ValueError, match="finite"):
        DelayedEnergyCalibrator(_weights(), b=float("nan"))
    with pytest.raises(ValueError, match="projection"):
        DelayedEnergyCalibrator(_weights(), b=9.0)

    adapter = DelayedEnergyCalibrator(_weights())
    adapter.record_prediction(
        "event",
        [0.0, 1.0],
        prediction_index=0,
        feedback_available_at=1,
        label_authority="fixture",
    )
    with pytest.raises(ValueError, match="already"):
        adapter.record_prediction(
            "event",
            [0.0, 1.0],
            prediction_index=0,
            feedback_available_at=1,
            label_authority="fixture",
        )
    with pytest.raises(ValueError, match="after prediction"):
        adapter.record_prediction(
            "same-step",
            [0.0, 1.0],
            prediction_index=1,
            feedback_available_at=1,
            label_authority="fixture",
        )
    with pytest.raises(ValueError, match="authority"):
        adapter.record_prediction(
            "no-authority",
            [0.0, 1.0],
            prediction_index=1,
            feedback_available_at=2,
            label_authority="",
        )


def test_erasure_replays_remaining_commits_and_restart_rejects_tampering() -> None:
    """SCENARIO-REPORT-7397-AUTHORITY: erasure keeps other commits exact."""

    adapter = DelayedEnergyCalibrator(_weights())
    for index in range(2):
        adapter.record_prediction(
            f"event-{index}",
            [float(index), float(1 - index)],
            prediction_index=index,
            feedback_available_at=index + 2,
            label_authority="fixture",
        )
    adapter.commit_feedback("event-0", 0, visible_at=2)
    adapter.commit_feedback("event-1", 1, visible_at=3)
    adapter.erase_feedback("event-0")
    assert adapter.update_count == 1

    checkpoint = adapter.to_dict()
    bad_weights = deepcopy(checkpoint)
    bad_weights["weights_hash"] = "sha256:bad"
    with pytest.raises(ValueError, match="weight hash"):
        DelayedEnergyCalibrator.from_dict(bad_weights)
    bad_state = deepcopy(checkpoint)
    bad_state["state_hash"] = "sha256:bad"
    with pytest.raises(ValueError, match="state hash"):
        DelayedEnergyCalibrator.from_dict(bad_state)

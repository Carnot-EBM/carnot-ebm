"""Focused contracts for REQ-REPORT-7660 and its three scenarios."""

import math
from types import SimpleNamespace

import pytest

from carnot.reporting import experiment_7660_atom_energy as energy

from carnot.reporting.experiment_7660_atom_energy import (
    FEATURES,
    action,
    energy_probability,
    fit_heads,
    join_role,
    score,
)


def _feature(unit: str, arm: str = "original_source", **changes: object) -> dict:
    row = {
        "unit_id": unit,
        "role": "fit",
        "partition": "fit_optimization",
        "arm": arm,
        "original_source_sha256": f"hash-{unit}",
        "source_sha256": f"hash-{unit}",
        "checked_structural_propositions": 1,
        "scoped_contradictions": 0,
        "ambiguity": 0,
        "unknown_claims": 1,
        "denominator": 2,
        "source_atom_count": 4,
        "whole_answer_certified": False,
    }
    row.update(changes)
    return row


def _label(unit: str, value: int = 1, **changes: object) -> dict:
    row = {
        "component_hash": unit,
        "role": "fit",
        "learning_partition": "fit_optimization",
        "label": value,
        "raw_probability": 0.25,
        "training_allowed": True,
        "evaluator_only": False,
    }
    row.update(changes)
    return row


def test_custody_uses_independent_label_not_witness() -> None:
    """SCENARIO-REPORT-7660-CUSTODY: witness status cannot become a target."""
    paired = join_role([_feature("a")], [_label("a", 0)], "fit")
    assert paired[0]["label"] == 0
    assert paired[0]["probability"] == 0.25
    for change in ({"role": "tune"}, {"learning_partition": "fit_anchor"}, {"label": 2}):
        with pytest.raises(ValueError):
            join_role([_feature("a")], [_label("a", **change)], "fit")
    with pytest.raises(ValueError):
        join_role([_feature("a", source_sha256="wrong")], [_label("a")], "fit")
    with pytest.raises(ValueError):
        join_role([_feature("a")], [_label("a", training_allowed=False)], "fit")


def test_energy_identity_extremes_and_metadata() -> None:
    """SCENARIO-REPORT-7660-ENERGY: finite normalization and metadata invariance."""
    identity = {
        "schema": "carnot.exp7660.head.v1",
        "feature_order": list(FEATURES),
        "weights": [0.0] * len(FEATURES),
        "clip": 1e-6,
        "normalization": "log1p_counts_and_sentence_fraction",
    }
    row = _feature("a")
    for p in (0.0, 0.25, 1.0):
        correct, error = energy_probability(row, p, identity)
        assert math.isfinite(correct) and math.isfinite(error)
        assert correct + error == pytest.approx(1.0)
        assert error == pytest.approx(min(1 - 1e-6, max(1e-6, p)))
    altered = {**row, "arbitrary_note": "irrelevant", "historically_exposed": False}
    assert score(row, 0.25, identity) == score(altered, 0.25, identity)
    assert action(0.25, (0.1, 0.9)) == action(score(row, 0.25, identity), (0.1, 0.9))
    with pytest.raises(ValueError):
        score(row, 0.25, {**identity, "feature_order": ["wrong"]})
    with pytest.raises(ValueError):
        score(row, float("inf"), identity)


def test_fit_preserves_identity_and_controls() -> None:
    """SCENARIO-REPORT-7660-REPLAY: heads freeze with source-erased control."""
    fit = [{"feature": _feature(str(i)), "label": i % 2, "probability": 0.25} for i in range(8)]
    tune = [
        {"feature": _feature(str(i)), "label": i % 2, "probability": 0.25} for i in range(8, 12)
    ]
    bundle = fit_heads(fit, tune)
    assert set(bundle["heads"]) == {"identity", "scalar", "cheap_atom", "atom", "source_erased"}
    assert len(bundle["settings"]) <= 6
    assert bundle["selected"] in bundle["heads"]
    assert bundle["heads"]["identity"]["weights"] == [0.0] * len(FEATURES)


def test_custody_rejects_missing_duplicate_and_invalid_inputs() -> None:
    """SCENARIO-REPORT-7660-CUSTODY: reject mismatched independent units."""
    with pytest.raises(ValueError, match="roster"):
        join_role([_feature("a")], [_label("a"), _label("a")], "fit")
    with pytest.raises(ValueError, match="arm"):
        join_role([_feature("a", arm="evidence_erasure")], [_label("a")], "fit")
    with pytest.raises(ValueError, match="missing"):
        join_role([_feature("a")], [_label("b")], "fit")
    with pytest.raises(ValueError, match="probability"):
        join_role([_feature("a")], [_label("a", raw_probability=float("nan"))], "fit")


def test_energy_rejects_bad_head_actions_and_empty_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    """SCENARIO-REPORT-7660-ENERGY: malformed head state cannot score."""
    head = energy._head([0.0] * len(FEATURES))
    with pytest.raises(ValueError, match="weight"):
        score(_feature("a"), 0.25, {**head, "weights": [float("nan"), 0, 0, 0]})
    with pytest.raises(ValueError, match="threshold"):
        action(0.5, (0.9, 0.1))
    assert action(0.01, (0.1, 0.9)) == "accept"
    assert action(0.99, (0.1, 0.9)) == "reject"
    with pytest.raises(ValueError, match="empty"):
        fit_heads([], [{"feature": _feature("a"), "label": 1, "probability": 0.25}])
    monkeypatch.setattr(energy, "minimize", lambda *args, **kwargs: SimpleNamespace(success=False))
    assert energy._fit([{"feature": _feature("a"), "label": 1, "probability": 0.25}], 1.0, (0,))[
        "weights"
    ] == [0.0] * len(FEATURES)

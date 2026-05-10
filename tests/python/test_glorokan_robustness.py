"""Tests for GloroKAN-style KArAt robustness bounds.

Spec references: REQ-KAN-1690, SCENARIO-KAN-1690.
"""

from __future__ import annotations

import json
from fractions import Fraction
from pathlib import Path

import pytest

from carnot.models.kan.glorokan_robustness import (
    GloroKANBounder,
    build_experiment_1690_artifact,
    write_experiment_1690_artifact,
)
from carnot.models.karat_attention import RationalKArAtLayer


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_req_kan_1690_spec_anchor_exists() -> None:
    """REQ-KAN-1690, SCENARIO-KAN-1690: robustness work is spec-anchored."""

    spec = (REPO_ROOT / "openspec/capabilities/kan/spec.md").read_text(encoding="utf-8")

    assert "REQ-KAN-1690" in spec
    assert "SCENARIO-KAN-1690" in spec
    assert "results/experiment_1690_glorokan_robustness.json" in spec


def test_req_kan_1690_bound_covers_same_radius_perturbation() -> None:
    """REQ-KAN-1690: local Lipschitz report bounds an observed energy delta."""

    layer = RationalKArAtLayer(seq_len=1, dim=2, spline_points=[0, 1, 2])
    bounder = GloroKANBounder(layer)
    q = [[Fraction(1, 4), Fraction(0)]]
    k = [[Fraction(1, 2), Fraction(0)]]
    radius = Fraction(1, 16)

    report = bounder.bound_forward(q, k, radius=radius)
    perturbed_q = [[Fraction(5, 16), Fraction(0)]]
    perturbed_k = [[Fraction(9, 16), Fraction(0)]]
    observed_delta = abs(layer.energy(perturbed_q, perturbed_k) - report.energy_at_center)

    assert report.norm == "linf"
    assert report.radius == radius
    assert report.local_lipschitz_bound == Fraction(1)
    assert report.energy_change_bound == Fraction(1, 16)
    assert observed_delta == Fraction(13, 256)
    assert observed_delta <= report.energy_change_bound

    serialized = report.as_serializable()
    assert serialized["local_lipschitz_bound"] == "1"
    assert serialized["energy_change_bound"] == "1/16"
    assert serialized["terms"][0]["dot_interval"] == ["5/64", "23/128"]
    assert serialized["terms"][0]["spline_slope_bound"] == "1"


def test_req_kan_1690_spline_slope_bounds_are_local_and_clamped() -> None:
    """REQ-KAN-1690: slope bounds use only reachable spline segments."""

    layer = RationalKArAtLayer(seq_len=1, dim=1, spline_points=[0, 1, 5])
    bounder = GloroKANBounder(layer)

    assert bounder.spline_slope_bound(Fraction(-3, 4), Fraction(-1, 4)) == Fraction(1)
    assert bounder.spline_slope_bound(Fraction(-1, 4), Fraction(-3, 4)) == Fraction(1)
    assert bounder.spline_slope_bound(Fraction(1, 4), Fraction(3, 4)) == Fraction(4)
    assert bounder.spline_slope_bound(Fraction(-1, 4), Fraction(1, 4)) == Fraction(4)
    assert bounder.spline_slope_bound(Fraction(3), Fraction(4)) == Fraction(0)


def test_req_kan_1690_validation_errors_are_explicit() -> None:
    """REQ-KAN-1690: malformed local-bound requests fail loudly."""

    layer = RationalKArAtLayer(seq_len=1, dim=2, spline_points=[0, 1])
    bounder = GloroKANBounder(layer)
    q = [[Fraction(0), Fraction(0)]]
    k = [[Fraction(0), Fraction(0)]]

    with pytest.raises(ValueError, match="radius must be nonnegative"):
        bounder.bound_forward(q, k, radius=Fraction(-1, 8))

    with pytest.raises(ValueError, match="only linf"):
        bounder.bound_forward(q, k, radius=Fraction(0), norm="l2")

    with pytest.raises(ValueError, match="q length"):
        bounder.bound_forward([], k, radius=Fraction(0))

    with pytest.raises(ValueError, match="k\\[0\\] dimension"):
        bounder.bound_forward(q, [[Fraction(0)]], radius=Fraction(0))


def test_scenario_kan_1690_builds_and_writes_artifact(tmp_path: Path) -> None:
    """SCENARIO-KAN-1690: artifact has stable schema fields."""

    artifact = build_experiment_1690_artifact()

    required_fields = {
        "schema",
        "status",
        "experiment",
        "experiment_id",
        "run_date",
        "spec",
        "module",
        "artifact_path",
        "local_lipschitz_bound",
        "energy_change_bound",
        "observed_witness_delta",
        "bound_covers_witness",
        "report",
        "honest_verdict",
    }
    assert required_fields <= set(artifact)
    assert artifact["schema"] == "carnot.glorokan_robustness.v1"
    assert artifact["status"] == "complete"
    assert artifact["experiment_id"] == 1690
    assert artifact["spec"] == ["REQ-KAN-1690", "SCENARIO-KAN-1690"]
    assert artifact["bound_covers_witness"] is True
    assert artifact["honest_verdict"] == "complete: glorokan_local_bounder_verified"

    output_path = tmp_path / "experiment_1690_glorokan_robustness.json"
    written = write_experiment_1690_artifact(output_path)

    assert written == artifact
    assert json.loads(output_path.read_text(encoding="utf-8")) == artifact

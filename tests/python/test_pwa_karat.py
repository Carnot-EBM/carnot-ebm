"""Tests for PWA KArAt Attention.

Spec references: REQ-KAN-1686, SCENARIO-KAN-1686.
"""

import json
from fractions import Fraction
from pathlib import Path

import pytest

from carnot.models.karat_attention import RationalKArAtLayer
from carnot.models.pwa_karat import (
    PWAKArAtAttention,
    build_experiment_1686_artifact,
    write_experiment_1686_artifact,
)


def test_pwa_karat_attention_evaluates_correctly() -> None:
    """REQ-KAN-1686: PWA KArAt Attention evaluates correctly."""
    layer = RationalKArAtLayer(seq_len=2, dim=2, spline_points=[Fraction(-1), Fraction(0), Fraction(1)])
    pwa = PWAKArAtAttention(layer, samples_per_segment=5)
    
    assert pwa.evaluate(-1.0) == pytest.approx(-1.0)
    assert pwa.evaluate(0.0) == pytest.approx(0.0)
    assert pwa.evaluate(1.0) == pytest.approx(1.0)
    assert pwa.evaluate(0.5) == pytest.approx(0.5)
    assert pwa.evaluate(-0.5) == pytest.approx(-0.5)


def test_build_experiment_1686_artifact() -> None:
    """SCENARIO-KAN-1686: Artifact contains required keys."""
    artifact = build_experiment_1686_artifact()
    assert artifact["schema"] == "carnot.pwa_karat_attention.v1"
    assert artifact["status"] == "complete"
    assert artifact["experiment"] == 1686
    assert artifact["honest_verdict"] == "complete: pwa_karat_attention_implemented"
    assert "pwa_unit" in artifact


def test_write_experiment_1686_artifact(tmp_path: Path) -> None:
    """SCENARIO-KAN-1686: Artifact is written to disk correctly."""
    out_path = tmp_path / "test_1686.json"
    written = write_experiment_1686_artifact(out_path)
    
    assert out_path.exists()
    content = json.loads(out_path.read_text(encoding="utf-8"))
    assert content["schema"] == "carnot.pwa_karat_attention.v1"
    assert content == written

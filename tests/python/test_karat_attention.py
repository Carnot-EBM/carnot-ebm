"""Test the KArAt attention block.

Traces to: REQ-KAN-1679, SCENARIO-KAN-1679.
"""

import json
from fractions import Fraction

import pytest

from carnot.models.karat_attention import (
    RationalKArAtLayer,
    build_experiment_1679_artifact,
    write_experiment_1679_artifact,
)


def test_karat_layer_initialization():
    layer = RationalKArAtLayer(seq_len=2, dim=3)
    assert layer.n_params == 3
    assert layer.seq_len == 2
    assert layer.dim == 3

    layer2 = RationalKArAtLayer(seq_len=2, dim=3, spline_points=[Fraction(-1), Fraction(1)])
    assert layer2.n_params == 2


def test_karat_layer_energy():
    layer = RationalKArAtLayer(seq_len=2, dim=2, spline_points=[Fraction(0), Fraction(1)])
    
    q = [[Fraction(1, 2), Fraction(0)], [Fraction(-1, 2), Fraction(1, 4)]]
    k = [[Fraction(0), Fraction(1)], [Fraction(1, 2), Fraction(-1, 4)]]
    
    # dot_products:
    # q[0]*k[0] = 0
    # q[0]*k[1] = 1/4
    # q[1]*k[0] = 1/4
    # q[1]*k[1] = -1/4 - 1/16 = -5/16
    
    # The spline evaluates points in [-1, 1]. Control points are at -1 and 1.
    # It interpolates between 0 and 1.
    # At x = -1 -> 0
    # At x = 1 -> 1
    # value = (x - (-1)) / 2 = (x + 1) / 2
    
    e00 = layer.attention_spline.evaluate(Fraction(0))  # 1/2
    e01 = layer.attention_spline.evaluate(Fraction(1, 4))  # (1.25)/2 = 5/8
    e10 = layer.attention_spline.evaluate(Fraction(1, 4))  # 5/8
    e11 = layer.attention_spline.evaluate(Fraction(-5, 16)) # (-5/16 + 1)/2 = (11/16)/2 = 11/32
    
    expected_energy = e00 + e01 + e10 + e11
    
    energy = layer.energy(q, k)
    assert energy == expected_energy


def test_karat_layer_bounds():
    layer = RationalKArAtLayer(seq_len=2, dim=2, spline_points=[Fraction(-2), Fraction(2)])
    min_b, max_b = layer.verify_bounding_bounds()
    assert min_b == Fraction(-8)  # -2 * 4
    assert max_b == Fraction(8)   # 2 * 4


def test_karat_layer_errors():
    layer = RationalKArAtLayer(seq_len=2, dim=2)
    q = [[Fraction(0), Fraction(0)]]  # Wrong seq_len
    k = [[Fraction(0), Fraction(0)], [Fraction(0), Fraction(0)]]
    with pytest.raises(ValueError, match="Sequence length mismatch"):
        layer.energy(q, k)
        
    q = [[Fraction(0), Fraction(0), Fraction(0)], [Fraction(0), Fraction(0)]] # Wrong dim in q
    with pytest.raises(ValueError, match="Dimension mismatch in q"):
        layer.energy(q, k)
        
    q = [[Fraction(0), Fraction(0)], [Fraction(0), Fraction(0)]]
    k = [[Fraction(0), Fraction(0), Fraction(0)], [Fraction(0), Fraction(0)]] # Wrong dim in k
    with pytest.raises(ValueError, match="Dimension mismatch in k"):
        layer.energy(q, k)


def test_build_experiment_artifact():
    artifact = build_experiment_1679_artifact()
    assert artifact["schema"] == "carnot.karat_attention.v1"
    assert artifact["status"] == "complete"
    assert artifact["experiment"] == 1679
    assert artifact["n_params"] == 3
    assert artifact["bounding_bounds_verified"] is True
    assert artifact["min_energy_bound"] == "-4"
    assert artifact["max_energy_bound"] == "4"
    assert "honest_verdict" in artifact


def test_write_experiment_artifact(tmp_path):
    out_path = tmp_path / "test_1679.json"
    write_experiment_1679_artifact(out_path)
    
    assert out_path.exists()
    content = json.loads(out_path.read_text())
    assert content["schema"] == "carnot.karat_attention.v1"

"""Tests for KANSymbolizer.

Verifies REQ-KAN-2060.
"""

import pytest
import numpy as np
import jax.numpy as jnp
from carnot.verify.kan_symbolizer import KANSymbolizer


class MockUnivariateKAEMLayer:
    def __init__(self, n_vars=2, n_knots=3):
        self.n_vars = n_vars
        self.n_knots = n_knots
        self._knots = jnp.array([-1.0, 0.0, 1.0])
        # ctrl points: var0 -> [0.0, 1.0, 2.0], var1 -> [1.0, 1.0, 1.0]
        self.control_points = jnp.array([[0.0, 1.0, 2.0], [1.0, 1.0, 1.0]])


class MockUnivariateKAEMLayerAlt:
    def __init__(self, n_vars=1, n_knots=2):
        self.n_vars = n_vars
        self.n_knots = n_knots
        self.knots = jnp.array([-1.0, 1.0])  # Uses knots instead of _knots
        self.control_points = jnp.array([[0.0, 1.0]])


class MockBadLayerNoVars:
    pass


class MockBadLayerNoKnots:
    def __init__(self):
        self.n_vars = 1
        self.n_knots = 2


class MockBadLayerNoControlPoints:
    def __init__(self):
        self.n_vars = 1
        self.n_knots = 2
        self._knots = jnp.array([-1.0, 1.0])


def test_kan_symbolizer_init_valid():
    """Test REQ-KAN-2060: initialize valid layer."""
    layer = MockUnivariateKAEMLayer()
    symbolizer = KANSymbolizer(layer)
    assert symbolizer.n_vars == 2
    assert symbolizer.n_knots == 3


def test_kan_symbolizer_init_invalid():
    """Test REQ-KAN-2060: missing properties raises error."""
    with pytest.raises(ValueError, match="must have n_vars and n_knots"):
        KANSymbolizer(MockBadLayerNoVars())


def test_kan_symbolizer_extract_invalid_knots():
    """Test REQ-KAN-2060: missing _knots raises error."""
    layer = MockBadLayerNoKnots()
    symbolizer = KANSymbolizer(layer)
    with pytest.raises(ValueError, match="must have _knots or knots property"):
        symbolizer.extract_piecewise_polynomials()


def test_kan_symbolizer_extract_invalid_ctrl():
    """Test REQ-KAN-2060: missing control_points raises error."""
    layer = MockBadLayerNoControlPoints()
    symbolizer = KANSymbolizer(layer)
    with pytest.raises(ValueError, match="must have control_points property"):
        symbolizer.extract_piecewise_polynomials()


def test_kan_symbolizer_extract_valid():
    """Test REQ-KAN-2060: extract knots and coeffs."""
    layer = MockUnivariateKAEMLayer()
    symbolizer = KANSymbolizer(layer)
    polys = symbolizer.extract_piecewise_polynomials()

    assert len(polys) == 2
    assert len(polys[0]) == 2  # 2 segments

    # var0 segment 0: [-1.0, 0.0], ctrl [0.0, 1.0] -> slope=1.0, intercept=1.0
    seg0 = polys[0][0]
    assert np.allclose(seg0["interval"], [-1.0, 0.0])
    assert np.allclose(seg0["coeffs"][1], 1.0)  # slope
    assert np.allclose(seg0["coeffs"][0], 1.0)  # intercept

    # var1 segment 0: [-1.0, 0.0], ctrl [1.0, 1.0] -> slope=0.0, intercept=1.0
    seg1 = polys[1][0]
    assert np.allclose(seg1["interval"], [-1.0, 0.0])
    assert np.allclose(seg1["coeffs"][1], 0.0)  # slope
    assert np.allclose(seg1["coeffs"][0], 1.0)  # intercept


def test_kan_symbolizer_extract_valid_alt():
    """Test REQ-KAN-2060: extract knots and coeffs using `knots`."""
    layer = MockUnivariateKAEMLayerAlt()
    symbolizer = KANSymbolizer(layer)
    polys = symbolizer.extract_piecewise_polynomials()
    assert len(polys) == 1


def test_kan_symbolizer_zero_dx():
    """Test REQ-KAN-2060: handle zero width interval."""
    layer = MockUnivariateKAEMLayer()
    layer._knots = jnp.array([-1.0, -1.0, 1.0])
    symbolizer = KANSymbolizer(layer)
    polys = symbolizer.extract_piecewise_polynomials()
    seg0 = polys[0][0]
    assert seg0["coeffs"][1] == 0.0  # slope should be 0 when dx=0


def test_kan_symbolizer_to_ast_string():
    """Test REQ-KAN-2060: generate AST string."""
    layer = MockUnivariateKAEMLayer()
    symbolizer = KANSymbolizer(layer)
    ast = symbolizer.to_ast_string()

    # It should look something like:
    # (+ (ite (< x_0 0.0) (+ 1.0 (* 1.0 x_0)) (+ 2.0 (* 1.0 x_0))) (ite ...))
    assert "(+" in ast
    assert "ite" in ast
    assert "x_0" in ast
    assert "x_1" in ast


def test_kan_symbolizer_to_ast_string_single_var():
    """Test REQ-KAN-2060: single variable AST string."""
    layer = MockUnivariateKAEMLayer(n_vars=1, n_knots=2)
    layer._knots = jnp.array([-1.0, 1.0])
    layer.control_points = jnp.array([[0.0, 1.0]])
    symbolizer = KANSymbolizer(layer)
    ast = symbolizer.to_ast_string()

    assert "ite" not in ast
    assert "(+ 0.5 (* 0.5 x_0))" == ast

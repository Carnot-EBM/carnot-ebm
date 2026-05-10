"""Tests for FourierCSP-boundary CIKAN verifier.

Spec: REQ-KAN-1723, SCENARIO-KAN-1723.
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from carnot.models.cikan_verifier import CIKAN, CIKANBoundary
from carnot.pipeline.fouriercsp_extractor import MultilinearPolynomial


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_req_kan_1723_spec_anchor_exists() -> None:
    """REQ-KAN-1723, SCENARIO-KAN-1723: the verifier is spec-anchored."""

    spec = (REPO_ROOT / "openspec/capabilities/kan/spec.md").read_text(encoding="utf-8")

    assert "REQ-KAN-1723" in spec
    assert "SCENARIO-KAN-1723" in spec
    assert "python/carnot/models/cikan_verifier.py" in spec
    assert "results/experiment_1723_cikan.json" in spec


def test_req_kan_1723_compiles_fouriercsp_boundary_as_fixed_architecture() -> None:
    """REQ-KAN-1723: FourierCSP constraints become immutable CIKAN boundaries."""

    constraint = MultilinearPolynomial(
        variables=["X", "Y"],
        expression="X AND Y",
        polynomial="X*Y",
    )
    model = CIKAN.from_fouriercsp(
        feature_names=["X", "Y"],
        constraints=[constraint],
        boundary_penalty=3.0,
        seed=7,
    )

    snapshot = model.boundary_snapshot()

    assert len(model.boundaries) == 1
    assert isinstance(model.boundaries[0], CIKANBoundary)
    assert snapshot == [
        {
            "name": "constraint_0",
            "variables": ["X", "Y"],
            "expression": "X AND Y",
            "polynomial": "X*Y",
            "variable_indices": [0, 1],
            "penalty": 3.0,
            "threshold": 0.5,
        }
    ]
    assert model.boundary_violations([1.0, 1.0]).tolist() == [0.0]
    assert model.boundary_violations([1.0, 0.0]).tolist() == [1.0]
    assert model.energy([1.0, 0.0]) > model.energy([1.0, 1.0])

    with pytest.raises(FrozenInstanceError):
        model.boundaries[0].expression = "X OR Y"  # type: ignore[misc]

    with pytest.raises(ValueError, match="unknown FourierCSP variable"):
        CIKAN.from_fouriercsp(
            feature_names=["X"],
            constraints=[constraint],
        )


def test_scenario_kan_1723_training_preserves_boundaries_and_energy_order() -> None:
    """SCENARIO-KAN-1723: training cannot move fixed FourierCSP boundaries."""

    constraint = MultilinearPolynomial(
        variables=["X", "Y"],
        expression="X AND Y",
        polynomial="X*Y",
    )
    model = CIKAN.from_fouriercsp(
        feature_names=["X", "Y"],
        constraints=[constraint],
        boundary_penalty=4.0,
        learning_rate=0.2,
        seed=11,
    )
    before = model.boundary_snapshot()
    x_train = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ],
        dtype=np.float32,
    )
    y_train = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    history = model.fit(x_train, y_train, epochs=50)
    metrics = model.evaluate(x_train, y_train)

    assert model.boundary_snapshot() == before
    assert history[0] > history[-1]
    assert metrics["accuracy"] == 1.0
    assert metrics["energy_gap"] > 0.0
    assert model.energy([1.0, 0.0]) > model.energy([1.0, 1.0])


def test_req_kan_1723_boolean_boundary_parser_supports_fouriercsp_operators() -> None:
    """REQ-KAN-1723: boundary units evaluate AND, OR, NOT, and XOR."""

    constraint = MultilinearPolynomial(
        variables=["A", "B", "C"],
        expression="(A OR NOT B) XOR C",
        polynomial="A + (1-B) + C - 2*(A + (1-B))*C",
    )
    model = CIKAN.from_fouriercsp(
        feature_names=["A", "B", "C"],
        constraints=[constraint],
        boundary_penalty=2.0,
    )

    assert model.boundary_violations([1.0, 1.0, 0.0]).tolist() == [0.0]
    assert model.boundary_violations([1.0, 1.0, 1.0]).tolist() == [1.0]

    with pytest.raises(ValueError, match="unsupported token"):
        CIKAN.from_fouriercsp(
            feature_names=["A"],
            constraints=[
                MultilinearPolynomial(
                    variables=["A"],
                    expression="A NAND A",
                    polynomial="bad",
                )
            ],
        ).boundary_violations([1.0])


def test_req_kan_1723_validation_and_parser_error_paths() -> None:
    """REQ-KAN-1723: invalid CIKAN and FourierCSP inputs fail explicitly."""

    with pytest.raises(ValueError, match="at least one feature"):
        CIKAN(feature_names=[])
    with pytest.raises(ValueError, match="unique"):
        CIKAN(feature_names=["A", "A"])
    with pytest.raises(ValueError, match="positive"):
        CIKAN(feature_names=["A"], boundary_penalty=0.0)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        CIKAN(feature_names=["A"], threshold=2.0)
    with pytest.raises(ValueError, match="n_knots"):
        CIKAN(feature_names=["A"], n_knots=1)
    with pytest.raises(ValueError, match="learning_rate"):
        CIKAN(feature_names=["A"], learning_rate=0.0)

    mapping_constraint = {
        "variables": ["A"],
        "expression": "A",
        "polynomial": "A",
    }
    model = CIKAN.from_fouriercsp(feature_names=["A"], constraints=[mapping_constraint])
    assert model.boundary_violations([1.0]).tolist() == [0.0]

    bad_expressions = [
        ("(A", "unexpected end"),
        ("(A B)", "missing closing"),
        ("AND A", "unsupported token"),
        ("A AND Z", "unknown FourierCSP variable"),
        ("A && B", "unsupported token near"),
        ("", "cannot be empty"),
    ]
    for expression, message in bad_expressions:
        with pytest.raises(ValueError, match=message):
            CIKAN.from_fouriercsp(
                feature_names=["A", "B"],
                constraints=[
                    {
                        "variables": ["A", "B"],
                        "expression": expression,
                        "polynomial": "bad",
                    }
                ],
            )

    with pytest.raises(ValueError, match="epochs"):
        model.fit([[1.0]], [1.0], epochs=0)
    with pytest.raises(ValueError, match="one label"):
        model.fit([[1.0]], [1.0, 0.0])
    with pytest.raises(ValueError, match="one label"):
        model.evaluate([[1.0]], [1.0, 0.0])
    with pytest.raises(ValueError, match="sample must have shape"):
        model.energy([1.0, 0.0])
    with pytest.raises(ValueError, match="batch must have shape"):
        model.energy_batch([1.0])

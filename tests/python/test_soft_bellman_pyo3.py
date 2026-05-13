"""PyO3 tests for the Soft Bellman ARM-to-EBM solver.

Spec coverage: REQ-INFER-2056, SCENARIO-INFER-2056-001,
SCENARIO-INFER-2056-002
"""

from __future__ import annotations

import math

import pytest

carnot_rust = pytest.importorskip(
    "carnot._rust",
    reason="Rust extension not built. Run: maturin develop -p crates/carnot-python",
)


def test_soft_bellman_solve_maps_logprobs_to_ebm_energy() -> None:
    """SCENARIO-INFER-2056-001: token energies are negated ARM logprobs."""
    result = carnot_rust.soft_bellman_solve([-0.2, -1.0, -0.05])

    assert result["immediate_rewards"] == pytest.approx([-0.2, -1.0, -0.05], abs=1e-7)
    assert result["token_energies"] == pytest.approx([0.2, 1.0, 0.05], abs=1e-7)
    assert result["soft_values"] == pytest.approx([0.0, 0.0, 0.0, 0.0], abs=1e-7)
    assert result["sequence_energy"] == pytest.approx(1.25, abs=1e-7)
    assert result["sequence_affinity"] == pytest.approx(-1.25, abs=1e-7)
    assert result["log_probability"] == pytest.approx(-1.25, abs=1e-7)
    assert result["log_partition"] == pytest.approx(0.0, abs=1e-7)


def test_soft_bellman_solve_accepts_logprob_rows_with_token_ids() -> None:
    """REQ-INFER-2056: PyO3 extracts chosen ARM logprobs from normalized rows."""
    rows = [[math.log(0.6), math.log(0.4)], [math.log(0.8), math.log(0.2)]]

    result = carnot_rust.soft_bellman_solve(rows, [1, 0])

    assert result["immediate_rewards"] == pytest.approx([math.log(0.4), math.log(0.8)])
    assert result["sequence_energy"] == pytest.approx(-(math.log(0.4) + math.log(0.8)))
    assert result["max_abs_bellman_residual"] <= 1e-6


def test_soft_bellman_solve_rejects_invalid_logprobs() -> None:
    """REQ-INFER-2056: invalid logprobs raise ValueError through PyO3."""
    with pytest.raises(ValueError, match="positive"):
        carnot_rust.soft_bellman_solve([-0.2, 0.1])

    with pytest.raises(ValueError, match="finite"):
        carnot_rust.soft_bellman_solve([-0.2, math.nan])


def test_soft_bellman_exhaustive_path_mass_is_normalized() -> None:
    """SCENARIO-INFER-2056-002: exp(-energy) recovers ARM path mass."""
    rows = [[math.log(0.75), math.log(0.25)], [math.log(0.4), math.log(0.6)]]
    masses = []

    for first in range(2):
        for second in range(2):
            result = carnot_rust.soft_bellman_solve([rows[0][first], rows[1][second]])
            masses.append(math.exp(-result["sequence_energy"]))

    assert sum(masses) == pytest.approx(1.0, abs=1e-7)

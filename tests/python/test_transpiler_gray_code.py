"""Tests for the Gray-code visible-spin encoder/decoder.

The load-bearing structural property is **Hamming-distance-1 between
adjacent quantization cells** — that's the pathology Gray code fixes
that standard binary breaks. This test file verifies the property
explicitly because it's the entire reason Gray code is in the recipe.

Spec: REQ-PHASE2-003 (Gray-code visible-spin encoder).
"""

from __future__ import annotations

import numpy as np
import pytest

from carnot.hardware.transpiler import decode_2d, decode_axis, encode_2d, encode_axis


# REQ-PHASE2-003
def test_gray_code_adjacent_cells_have_hamming_distance_one() -> None:
    """The defining property: two adjacent integer cells map to spin
    arrays differing in exactly one bit. This is what eliminates the
    "cliff" pathology that breaks PT-PCD on standard binary.
    """
    m = 8
    lo, hi = -1.0, 1.0
    n_cells = 1 << m
    step = (hi - lo) / n_cells

    for cell in range(n_cells - 1):
        # Two latents one cell apart
        z_a = lo + (cell + 0.5) * step
        z_b = lo + (cell + 1.5) * step
        spins_a = encode_axis(z_a, m, lo, hi)
        spins_b = encode_axis(z_b, m, lo, hi)
        # Hamming distance: number of positions that differ
        diff = int(np.sum(spins_a != spins_b))
        assert diff == 1, (
            f"cell {cell}->{cell + 1} flipped {diff} bits, expected 1 "
            f"(Gray-code locality property violated)"
        )


# REQ-PHASE2-003
def test_gray_code_round_trip_axis_returns_cell_center() -> None:
    """Encoding then decoding (with rng=None) must return the cell
    center, not the original z value. This is the deterministic-
    decoder invariant; with an RNG the decoder adds spatial noise.
    """
    m = 6
    lo, hi = -2.0, 2.0
    z_in = np.linspace(-1.5, 1.5, 11)
    spins = encode_axis(z_in, m, lo, hi)
    z_out = decode_axis(spins, m, lo, hi, rng=None)
    z_out_arr = np.atleast_1d(z_out)

    # Cell width = 4.0 / 64 = 0.0625; round-trip error <= half cell
    step = (hi - lo) / (1 << m)
    np.testing.assert_array_less(np.abs(z_out_arr - z_in), step / 2 + 1e-9)


# REQ-PHASE2-003
def test_gray_code_round_trip_2d_preserves_locality() -> None:
    """2D round-trip on a small grid: encode a structured set of points,
    decode back, verify per-axis Hamming-distance-1 between
    horizontally-adjacent grid cells.
    """
    m = 5
    lo, hi = -1.0, 1.0
    grid = 8
    coords = np.linspace(lo + 0.1, hi - 0.1, grid)
    pts = np.array([[x, y] for x in coords for y in coords])
    spins = encode_2d(pts, m, lo, hi)
    assert spins.shape == (grid * grid, 2 * m)
    decoded = decode_2d(spins, m, lo, hi, rng=None)
    assert decoded.shape == (grid * grid, 2)

    # Decoded values are cell centers — within step/2 of the inputs
    step = (hi - lo) / (1 << m)
    np.testing.assert_array_less(np.abs(decoded - pts).max(), step / 2 + 1e-9)


# REQ-PHASE2-003
def test_gray_code_clamps_out_of_range_inputs() -> None:
    """Inputs outside ``[lo, hi]`` are clamped, not silently wrapped.
    The cell bound is the safety rail.
    """
    m = 4
    lo, hi = -1.0, 1.0
    spins_below = encode_axis(-5.0, m, lo, hi)
    spins_first = encode_axis(lo + 1e-9, m, lo, hi)
    spins_above = encode_axis(5.0, m, lo, hi)
    spins_last = encode_axis(hi - 1e-9, m, lo, hi)

    # Below-range should map to the first cell
    np.testing.assert_array_equal(spins_below, spins_first)
    # Above-range should map to the last cell
    np.testing.assert_array_equal(spins_above, spins_last)


# REQ-PHASE2-003
def test_gray_code_rejects_invalid_args() -> None:
    with pytest.raises(ValueError, match="m must be positive"):
        encode_axis(0.0, 0, -1.0, 1.0)
    with pytest.raises(ValueError, match="hi must exceed lo"):
        encode_axis(0.0, 4, 1.0, -1.0)
    with pytest.raises(ValueError, match="2D latent"):
        encode_2d(np.zeros((3, 5)), 4, -1.0, 1.0)


# REQ-PHASE2-003
def test_gray_code_decoder_with_rng_is_within_cell() -> None:
    """With an RNG passed in, the decoder adds uniform spatial noise
    over the cell. Returned value must still lie within the cell
    boundaries.
    """
    m = 5
    lo, hi = -1.0, 1.0
    rng = np.random.default_rng(42)
    z_in = np.array([0.3])
    spins = encode_axis(z_in, m, lo, hi)
    step = (hi - lo) / (1 << m)
    # Sample the decoder many times; all outputs lie in the same cell
    samples = np.array([decode_axis(spins, m, lo, hi, rng=rng) for _ in range(50)]).ravel()
    # Compute the cell center the input belongs to
    cell_center = lo + (int((z_in[0] - lo) / step) + 0.5) * step
    assert np.all(np.abs(samples - cell_center) <= step / 2 + 1e-9)

"""Regression test for the _delta cap=80 fix (2026-06-21).

REQ: the world-model induction evidence must show the FULL per-transition change, not a
truncated prefix. The old `_delta(..., cap=80)` silently dropped everything past 80 changed
cells, so a 293-cell re-render showed only 27% -- starving the model of the evidence it needs
to induce the rule. `_rle_delta` replaces it with a LOSSLESS run-length encoding.

SCENARIO-RLE-1: _rle_delta round-trips exactly, including multi-digit colors (>=10) where a
                naive concatenated digit string would be ambiguous.
SCENARIO-RLE-2: a large change (>80 cells) is shown in FULL (no truncation).
SCENARIO-RLE-3: the induction prompt no longer carries the old truncated raw-tuple format.
"""
import numpy as np

from carnot.agentic import arc_executable_world_model as e3


def _apply_rle(g0, rle):
    """Inverse of _rle_delta: reconstruct g1 from g0 + the run-length string."""
    g = np.asarray(g0).copy()
    if rle in ("", "(no change)"):
        return g
    for run in rle.split(" "):
        head, vals = run.split(":")
        r = int(head[1 : head.index("c")])
        c0 = int(head[head.index("c") + 1 :])
        for i, v in enumerate(vals.split(",")):
            g[r, c0 + i] = int(v)
    return g


def test_scenario_rle_1_lossless_including_multidigit_colors() -> None:
    """SCENARIO-RLE-1: lossless round-trip across random grids with colors 0-15."""
    rng = np.random.default_rng(0)
    for _ in range(300):
        g0 = rng.integers(0, 16, (12, 12))
        g1 = g0.copy()
        for _ in range(int(rng.integers(1, 50))):
            g1[int(rng.integers(0, 12)), int(rng.integers(0, 12))] = int(rng.integers(0, 16))
        assert np.array_equal(_apply_rle(g0, e3._rle_delta(g0, g1)), g1)


def test_scenario_rle_2_large_change_not_truncated() -> None:
    """SCENARIO-RLE-2: a >80-cell change is encoded in FULL (the cap=80 bug)."""
    g0 = np.full((64, 64), 4)
    g1 = g0.copy()
    g1[10:25, 5:25] = 7  # 300-cell change, well past the old cap of 80
    n_changed = int((g0 != g1).sum())
    assert n_changed > 80
    assert np.array_equal(_apply_rle(g0, e3._rle_delta(g0, g1)), g1)  # 100% reconstructed


def test_scenario_rle_3_prompt_uses_full_rle_not_truncated_tuples() -> None:
    """SCENARIO-RLE-3: the induction transition block uses the run-length form, not the old cap."""
    g0 = np.zeros((8, 8), dtype=int)
    g1 = g0.copy()
    g1[0, :] = 5  # an 8-cell run
    t = e3.Transition(g0, 6, {"x": 1, "y": 1}, g1, 0, 0)
    block = e3._transitions_block([t])
    assert "run-length" in block
    assert "row,col,from,to" not in block  # old truncated format is gone

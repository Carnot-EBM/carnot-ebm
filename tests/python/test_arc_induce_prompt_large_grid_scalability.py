"""Tests for the induce_prompt large-grid scalability fix (task 11's second half).

Spec refs: REQ-ARC-WMTE-5593-2, SCENARIO-ARC-WMTE-5593-2-LOSSLESS-ROUND-TRIP,
SCENARIO-ARC-WMTE-5593-2-REAL-BUDGET-FIT.
"""

from __future__ import annotations

import numpy as np

from carnot.agentic import arc_executable_world_model as e3


def _reconstruct_grid(rle: str, shape: tuple[int, int]) -> np.ndarray:
    """Inverse of `_rle_grid`: implicit column, one line per row."""

    g = np.zeros(shape, dtype=int)
    for line in rle.split("\n"):
        head, rest = line.split(":", 1)
        r = int(head[1:])
        c = 0
        for run in rest.split(","):
            v, n = run.split("x")
            n = int(n)
            g[r, c : c + n] = int(v)
            c += n
    return g


def _reconstruct_delta(g0: np.ndarray, rle: str) -> np.ndarray:
    """Inverse of `_rle_delta_compact`: explicit run-start column, implicit sub-run column."""

    g = np.asarray(g0).copy()
    if rle in ("", "(no change)"):
        return g
    for run in rle.split(" "):
        head, rest = run.split(":", 1)
        r = int(head[1 : head.index("c")])
        c = int(head[head.index("c") + 1 :])
        for pair in rest.split(","):
            v, n = pair.split("x")
            n = int(n)
            g[r, c : c + n] = int(v)
            c += n
    return g


def test_scenario_5593_2_rle_grid_lossless_round_trip_random() -> None:
    """SCENARIO-ARC-WMTE-5593-2-LOSSLESS-ROUND-TRIP: randomized grids round-trip exactly."""

    rng = np.random.default_rng(1)
    for _ in range(200):
        h, w = int(rng.integers(1, 20)), int(rng.integers(1, 20))
        g = rng.integers(0, 16, (h, w))
        rle = e3._rle_grid(g)
        assert np.array_equal(_reconstruct_grid(rle, g.shape), g)


def test_scenario_5593_2_rle_grid_uniform_row_collapses_to_one_run() -> None:
    """A uniform 64-cell row collapses to a single run, not 64 characters."""

    g = np.full((1, 64), 3)
    rle = e3._rle_grid(g)
    assert rle == "r0:3x64"


def test_scenario_5593_2_rle_delta_compact_lossless_round_trip_random() -> None:
    """SCENARIO-ARC-WMTE-5593-2-LOSSLESS-ROUND-TRIP: randomized diffs round-trip exactly,
    including multi-digit colors (>=10) and runs with repeated new values."""

    rng = np.random.default_rng(2)
    for _ in range(200):
        g0 = rng.integers(0, 16, (12, 12))
        g1 = g0.copy()
        for _ in range(int(rng.integers(1, 50))):
            g1[int(rng.integers(0, 12)), int(rng.integers(0, 12))] = int(rng.integers(0, 16))
        rle = e3._rle_delta_compact(g0, g1)
        assert np.array_equal(_reconstruct_delta(g0, rle), g1)


def test_scenario_5593_2_rle_delta_compact_no_change_is_explicit() -> None:
    g0 = np.zeros((4, 4), dtype=int)
    assert e3._rle_delta_compact(g0, g0.copy()) == "(no change)"


def test_scenario_5593_2_rle_delta_compact_collapses_repeated_run_values() -> None:
    """A changed run whose new values are all identical collapses to one 'value x count' pair,
    not one token per cell -- the fix for the delta cost that became dominant after the
    full-grid fix landed."""

    g0 = np.zeros((1, 10), dtype=int)
    g1 = g0.copy()
    g1[0, 2:8] = 7  # 6-cell uniform-value change
    rle = e3._rle_delta_compact(g0, g1)
    assert rle == "r0c2:7x6"


def test_scenario_5593_2_transitions_block_uses_the_new_compact_encoders() -> None:
    """`_transitions_block` routes full grids through `_rle_grid` and deltas through
    `_rle_delta_compact` (not the raw `to_ascii` / verbose `_rle_delta`)."""

    g0 = np.zeros((8, 8), dtype=int)
    g1 = g0.copy()
    g1[0, :] = 5
    t = e3.Transition(g0, 6, {"x": 1, "y": 1}, g1, 0, 0)
    block = e3._transitions_block([t])
    assert "r0:0x8" in block  # INITIAL grid via _rle_grid, implicit column
    assert "r0c0:5x8" in block  # delta via _rle_delta_compact, collapsed run
    assert "run-length" in block


def test_req_arc_wmte_5593_2_spec_declares_scalability_fix() -> None:
    from pathlib import Path

    spec_path = (
        Path(__file__).resolve().parents[2]
        / "openspec"
        / "capabilities"
        / "arc-human-replay-frame-change"
        / "spec.md"
    )
    spec = spec_path.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-ARC-WMTE-5593-2") :]

    for marker in (
        "REQ-ARC-WMTE-5593-2",
        "SCENARIO-ARC-WMTE-5593-2-LOSSLESS-ROUND-TRIP",
        "SCENARIO-ARC-WMTE-5593-2-REAL-BUDGET-FIT",
        "11,167",
        "18,355",
    ):
        assert marker in section


# ---------------------------------------------------------------------------
# REQ-ARC-FCP-5699-23: DEV-ONLY override so the LLM can be shown more than the
# default ~6 grid-changing transitions -- REQ-ARC-FCP-5699-22 found the default
# starves the dynamics half to roughly one example per action type, producing
# hardcoded-literal-coordinate memorization instead of general rules (g50t).
# ---------------------------------------------------------------------------


def _transitions(n, *, shape=(8, 8)):
    out = []
    for i in range(n):
        g0 = np.zeros(shape, dtype=int)
        g1 = g0.copy()
        g1[0, i % shape[1]] = 5  # each transition changes a DIFFERENT cell -> distinguishable
        out.append(e3.Transition(g0, (i % 7) + 1, None, g1, 0, 0))
    return out


def test_req_arc_fcp_5699_23_induce_transitions_k_defaults_to_8_when_unset(monkeypatch):
    monkeypatch.delenv("CARNOT_ARC_INDUCE_TRANSITIONS_K", raising=False)
    assert e3._induce_transitions_k() == 8


def test_req_arc_fcp_5699_23_induce_transitions_k_env_override(monkeypatch):
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TRANSITIONS_K", "20")
    assert e3._induce_transitions_k() == 20


def test_req_arc_fcp_5699_23_induce_prompt_default_k_matches_pre_existing_behavior():
    """Regression-safety anchor: induce_prompt with NO explicit k arg (every pre-5699-23 call
    site) must show the exact same transitions as before -- k defaults to 8, byte-identical."""
    trans = _transitions(25)
    prompt_default = e3.induce_prompt("paritytest", trans, cell=1)
    prompt_explicit_8 = e3.induce_prompt("paritytest", trans, cell=1, k=8)
    assert prompt_default == prompt_explicit_8


def test_req_arc_fcp_5699_23_raising_k_shows_more_transitions_to_the_llm():
    """The actual fix under test: a higher k must surface MORE grid-changing transitions in
    the rendered prompt, out of a pool large enough that the cap is the binding constraint.
    All 25 synthetic transitions CHANGE the grid (no no-ops), so sample size = k - 2 (the
    `changed[:k-2] + noop[:2]` split in _transitions_block, with an empty noop pool here)."""
    trans = _transitions(25)  # 25, matching the real stall-trigger transition_count
    prompt_k8 = e3.induce_prompt("paritytest", trans, cell=1, k=8)
    prompt_k20 = e3.induce_prompt("paritytest", trans, cell=1, k=20)
    n_actions_k8 = prompt_k8.count("--- ACTION")
    n_actions_k20 = prompt_k20.count("--- ACTION")
    assert n_actions_k8 == 6  # k - 2, no no-ops in this synthetic set
    assert n_actions_k20 == 18
    assert n_actions_k20 > n_actions_k8

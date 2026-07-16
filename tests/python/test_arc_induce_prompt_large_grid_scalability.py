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


# ---------------------------------------------------------------------------
# REQ-ARC-FCP-5699-26: refactor_prompt's mismatch payload must be VALID JSON at any size.
# The pre-existing json.dumps(vr.mismatches[:5], indent=1)[:4000] hard-truncated the encoded
# string by raw character count, which can (and on a real g50t counterexample, does -- 5 real
# mismatches serialize to 12,212 chars) slice through the middle of a JSON structure, producing
# genuinely invalid JSON shown to the LLM as the thing it's meant to debug from.
# ---------------------------------------------------------------------------


def _mismatch(i, n_cells=30):
    cells = [[r, 0, 1, 2] for r in range(n_cells)]
    return {
        "i": i,
        "action": 1,
        "data": None,
        "true_change": list(cells),
        "your_prediction_was_wrong_at": list(cells),
    }


def test_req_arc_fcp_5699_26_bounded_mismatches_caps_large_cell_lists():
    bounded = e3._bounded_mismatches([_mismatch(0, n_cells=30)])
    assert len(bounded) == 1
    assert len(bounded[0]["true_change"]) == e3._REFACTOR_PROMPT_MAX_CELLS_PER_MISMATCH
    assert (
        bounded[0]["true_change_omitted_count"] == 30 - e3._REFACTOR_PROMPT_MAX_CELLS_PER_MISMATCH
    )
    assert (
        bounded[0]["your_prediction_was_wrong_at_omitted_count"]
        == 30 - e3._REFACTOR_PROMPT_MAX_CELLS_PER_MISMATCH
    )


def test_req_arc_fcp_5699_26_bounded_mismatches_leaves_small_lists_untouched():
    bounded = e3._bounded_mismatches([_mismatch(0, n_cells=3)])
    assert bounded[0]["true_change"] == [[r, 0, 1, 2] for r in range(3)]
    assert "true_change_omitted_count" not in bounded[0]
    assert "your_prediction_was_wrong_at_omitted_count" not in bounded[0]


def test_req_arc_fcp_5699_26_refactor_prompt_produces_valid_json_at_realistic_scale():
    """Regression test for the exact bug: 5 mismatches this large previously serialized past
    4000 chars and got hard-truncated into invalid JSON. Confirms the fix's MISMATCHES block
    parses as JSON regardless."""
    import json as _json
    from types import SimpleNamespace

    mismatches = [_mismatch(i, n_cells=30) for i in range(5)]
    # sanity-check the regression premise: the OLD approach really would have produced
    # invalid JSON at this scale.
    old_style = _json.dumps(mismatches[:5], indent=1)[:4000]
    broke = False
    try:
        _json.loads(old_style)
    except _json.JSONDecodeError:
        broke = True
    assert broke, "test fixture must reproduce the truncation bug to be a real regression test"

    vr = SimpleNamespace(n=25, n_correct=5, accuracy=0.2, mismatches=mismatches)
    prompt = e3.refactor_prompt("paritytest", vr)
    mism_block = prompt[prompt.index("MISMATCHES:\n") + len("MISMATCHES:\n") :]
    parsed = _json.loads(mism_block)  # must not raise
    assert len(parsed) == 5


# ---------------------------------------------------------------------------
# REQ-ARC-FCP-5699-31: DEV-ONLY structural reminder in refactor_prompt, targeting the exact
# pathology REQ-ARC-FCP-5699-30 found by reading a real raw completion: the model wrapped its
# fix in a class with self-bound methods, invented a fictional grid representation, and never
# emitted is_level_complete at all.
# ---------------------------------------------------------------------------


def _refactor_vr():
    from types import SimpleNamespace

    return SimpleNamespace(
        n=25,
        n_correct=5,
        accuracy=0.2,
        mismatches=[
            {
                "i": 0,
                "action": 1,
                "data": None,
                "true_change": [],
                "your_prediction_was_wrong_at": [],
            }
        ],
    )


def test_req_arc_fcp_5699_31_refactor_prompt_default_unset_is_byte_identical(monkeypatch):
    """Regression-safety anchor: CARNOT_ARC_REFACTOR_STRUCTURE_REMINDER unset (production
    default) -- the prompt must be byte-identical to the pre-5699-31 template."""
    monkeypatch.delenv("CARNOT_ARC_REFACTOR_STRUCTURE_REMINDER", raising=False)
    vr = _refactor_vr()
    prompt = e3.refactor_prompt("g50t", vr)
    assert "REQUIRED OUTPUT STRUCTURE" not in prompt
    assert "class WorldModel" not in prompt  # sanity: nothing about classes is injected
    tail = (
        "MISMATCHES:\n"
        + __import__("json").dumps(e3._bounded_mismatches(vr.mismatches), indent=1)
        + "\n"
    )
    assert prompt.endswith(tail)


def test_req_arc_fcp_5699_31_refactor_prompt_reminder_targets_the_observed_pathology(monkeypatch):
    """When enabled, the reminder must explicitly forbid the exact structure REQ-ARC-FCP-5699-30
    observed the model produce: a class, self-bound methods, and an invented grid shape."""
    monkeypatch.setenv("CARNOT_ARC_REFACTOR_STRUCTURE_REMINDER", "1")
    vr = _refactor_vr()
    prompt = e3.refactor_prompt("g50t", vr)
    assert "REQUIRED OUTPUT STRUCTURE" in prompt
    assert "def engine(grid, action, data):" in prompt
    assert "def is_level_complete(grid):" in prompt
    assert "Do NOT wrap them in a class" in prompt
    assert "Do NOT use `self`" in prompt
    assert "Do NOT invent a new internal grid" in prompt
    # the reminder appears both before (primacy) and after (recency, right before generation)
    # the mismatches block
    mismatches_idx = prompt.index("MISMATCHES:")
    assert prompt.index("REQUIRED OUTPUT STRUCTURE") < mismatches_idx
    assert prompt.rindex("Reminder:") > mismatches_idx

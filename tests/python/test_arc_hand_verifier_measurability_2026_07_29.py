"""SCENARIO-ARC-WMTE-6044-PROGRESS-ENDPOINT-DOES-NOT-REPORT-AN-INSTRUMENT-FLOOR.

`hv_progress` must not report an INSTRUMENT FLOOR as an observation.

THE INCIDENT (2026-07-29). `hv_progress` is `(start_hv - min_hv) / max(|start_hv|, 1)`, computed
from the per-game adapter `hand_verifier`. If that verifier returns THE SAME VALUE on every frame,
`hv_progress` is identically 0.0 for ANY run of that game -- no matter what the agent does. That is
the exact defect that invalidated the `accuracy` endpoint, where a positive control showed a total
answer leak could not move it off 0.0.

It is the common case, not a corner. An AST census of `arc_game_adapters.py` finds 4 of the 22
adaptered public games shipping a literal constant:

    cn04, ka59, sp80, su15   ->   hand_verifier=lambda _game, _frame=None: 0.0

plus 3 of the 25 survey games with no adapter at all: 7 of 25 immovable by construction. In the
24-cell retention A/B that was 8 cells (33%) whose 0.0 was pooled with real observations inside a
rank correlation, an entropy figure, a zero-count distribution and a discordance tally. Excluding
them moved the headline correlation from +0.0929 to -0.0040.

Note that `ka59` -- the control game used for the goal-gate acceptance test -- is one of the stubs.

TWO FURTHER PROPERTIES, previously invisible (same date):

  * Exceptions were SWALLOWED (`except Exception: return None`). A None on the FIRST call leaves
    `start_hv` unset, so the run silently rebases its own baseline to a later frame.
  * Several adapters return a 1000.0 "search stops here" sentinel. A sentinel on the first call pins
    `start_hv` at 1000.0, after which any ordinary value yields hv_progress ~= 1.0 FOR FREE --
    ceiling corruption that reads as near-perfect progress.
"""

import ast

import numpy as np

from carnot.agentic import arc_actions_to_progress as atp

# The literal-constant stubs, verified by AST census against arc_game_adapters.py. Asserted below
# rather than trusted, so this list cannot silently drift out of date.
_KNOWN_CONSTANT_STUBS = {"cn04", "ka59", "sp80", "su15"}


class _FakeFrame:
    """Minimal stand-in for an arcade frame: `grid_of` reads `.frame[-1]`-shaped payloads."""

    def __init__(self, grid: np.ndarray) -> None:
        self.frame = [grid.tolist()]


def _drive(game: str, grids: list[np.ndarray]) -> dict:
    fn = atp._hand_verifier_fn(game)
    assert fn is not None, (
        f"{game} must have an adapter hand_verifier for this test to mean anything"
    )
    for grid in grids:
        fn(None, _FakeFrame(grid))
    return fn.stats


def test_ast_census_still_finds_exactly_the_known_constant_stubs():
    """The measurability check exists because these stubs exist. If one is fixed (or a new one is
    added) this test fails, forcing the docstring and the census above to be updated rather than
    silently becoming wrong."""
    tree = ast.parse(open("python/carnot/agentic/arc_game_adapters.py").read())
    constant = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
        if name != "GameAdapter":
            continue
        game = hv = None
        for kw in node.keywords:
            if kw.arg == "game":
                game = getattr(kw.value, "value", None)
            elif kw.arg == "hand_verifier":
                hv = kw.value
        if game and isinstance(hv, ast.Lambda) and isinstance(hv.body, ast.Constant):
            constant.add(game)
    assert constant == _KNOWN_CONSTANT_STUBS, (
        f"the set of literal-constant hand_verifiers changed: {sorted(constant)}. "
        "Update _KNOWN_CONSTANT_STUBS and this module's docstring."
    )


def test_constant_verifier_over_two_distinct_frames_is_unmeasurable():
    """A single value across two DIFFERENT frames is an instrument floor, and must be reported as
    such -- this is the property `hv_progress_measurable` encodes.

    Asserts the RULE (`hv_progress_measurable_from_stats`), not merely the raw counters: a test that
    only inspects the stats dict passes even if the rule is loosened to always return True, which is
    precisely the mutation that must fail here.
    """
    stats = _drive("su15", [np.zeros((4, 4), dtype=int), np.ones((4, 4), dtype=int)])
    assert stats["n_calls"] == 2
    assert stats["n_exceptions"] == 0
    assert len(stats["distinct_values"]) == 1, "su15's stub returns one value by construction"
    assert len(stats["distinct_frame_keys"]) == 2, "two genuinely different frames were shown"
    assert atp.hv_progress_measurable_from_stats(stats) is False, (
        "a constant verifier over distinct frames must be UNMEASURABLE; reporting its 0.0 as an "
        "observation is the instrument-floor defect this check exists to stop"
    )


def test_one_value_over_one_frame_is_not_called_unmeasurable():
    """Guard against over-claiming in the other direction: a verifier that only ever saw ONE frame
    has produced no evidence about constancy, and declaring it unmeasurable would be its own false
    claim. The rule must return None (undecided), not False."""
    grid = np.zeros((4, 4), dtype=int)
    stats = _drive("su15", [grid, grid.copy()])  # same content twice
    assert len(stats["distinct_frame_keys"]) == 1
    assert len(stats["distinct_values"]) == 1
    assert atp.hv_progress_measurable_from_stats(stats) is None, (
        "one frame is no evidence about constancy; the rule must abstain, not assert False"
    )


def test_measurability_rule_table():
    """The whole rule, as an explicit table, so any future loosening fails a named case."""

    def stats(values, frames):
        return {"distinct_values": set(values), "distinct_frame_keys": set(frames), "n_calls": 9}

    # varying values -> measurable regardless of frame bookkeeping
    assert atp.hv_progress_measurable_from_stats(stats([0.0, 1.0], [1])) is True
    assert atp.hv_progress_measurable_from_stats(stats([0.0, 1.0], [1, 2])) is True
    # one value over several distinct frames -> the floor case
    assert atp.hv_progress_measurable_from_stats(stats([0.0], [1, 2])) is False
    assert atp.hv_progress_measurable_from_stats(stats([5.0], [1, 2, 3])) is False
    # insufficient evidence -> abstain
    assert atp.hv_progress_measurable_from_stats(stats([0.0], [1])) is None
    assert atp.hv_progress_measurable_from_stats(stats([], [])) is None
    # no verifier at all
    assert atp.hv_progress_measurable_from_stats(None) is None


def _stub_adapter_with(verifier):
    """Build a throwaway adapter carrying `verifier`.

    `get_adapter` calls the per-game BUILDER on every invocation, so mutating a returned adapter
    does nothing to the next lookup -- the registry must be patched, not the instance.
    """
    from carnot.agentic.arc_game_adapters import GameAdapter

    return GameAdapter(
        game="stub",
        action_labels=lambda _g: [],
        apply=lambda g, _l, _d=None: g,
        state_key=lambda _g: 0,
        hand_verifier=verifier,
    )


def test_hand_verifier_exceptions_are_counted_not_swallowed(monkeypatch):
    """A raising verifier must still not crash the episode (this is an instrument bolted to a live
    agent), but the failure must leave a record. Swallowing it silently is how a rebased baseline
    became invisible."""

    def _boom(*_args, **_kwargs):
        raise RuntimeError("verifier boom")

    from carnot.agentic import arc_game_adapters as adapters

    monkeypatch.setattr(adapters, "get_adapter", lambda _game: _stub_adapter_with(_boom))
    fn = atp._hand_verifier_fn("stub")
    assert fn is not None
    assert fn(None, _FakeFrame(np.zeros((4, 4), dtype=int))) is None
    assert fn.stats["n_exceptions"] == 1, "the exception must be counted, not vanish"


def test_sentinel_returns_are_counted(monkeypatch):
    """The 1000.0 'search stops here' sentinel pins `start_hv` when it lands on the first call,
    which makes any later ordinary value read as ~1.0 progress for free. Counting it lets a
    suspiciously high hv_progress be checked against the sentinel fraction."""
    from carnot.agentic import arc_game_adapters as adapters

    monkeypatch.setattr(
        adapters,
        "get_adapter",
        lambda _game: _stub_adapter_with(lambda _g, _frame=None: 1000.0),
    )
    fn = atp._hand_verifier_fn("stub")
    assert fn is not None
    fn(None, _FakeFrame(np.zeros((4, 4), dtype=int)))
    fn(None, _FakeFrame(np.ones((4, 4), dtype=int)))
    assert fn.stats["n_sentinel"] == 2
    assert fn.stats["n_calls"] == 2


def test_a_genuinely_varying_verifier_is_measurable(monkeypatch):
    """The positive control for the measurability check itself: a verifier that DOES vary must be
    classified measurable, otherwise the check would suppress every cell rather than the floored
    ones."""
    from carnot.agentic import arc_game_adapters as adapters

    values = iter([10.0, 7.0, 3.0])
    monkeypatch.setattr(
        adapters,
        "get_adapter",
        lambda _game: _stub_adapter_with(lambda _g, _frame=None: next(values)),
    )
    fn = atp._hand_verifier_fn("stub")
    assert fn is not None
    for i in range(3):
        fn(None, _FakeFrame(np.full((4, 4), i, dtype=int)))
    assert len(fn.stats["distinct_values"]) == 3
    assert len(fn.stats["distinct_frame_keys"]) == 3
    assert fn.stats["n_sentinel"] == 0


def test_progress_result_reports_measurability_fields():
    """The fields must exist on the dataclass and survive `to_row`, otherwise a downstream analysis
    cannot tell an unmeasurable cell from a measured zero -- which is the whole point."""
    row = atp.ProgressResult(
        game="su15",
        arm="a",
        seed=1,
        variant=0,
        start_level=0,
        reached_level=0,
        levels_gained=0,
        solved=False,
        actions_to_first_solve=None,
        total_actions=10,
        noop_frac=0.0,
        revisit_frac=0.0,
        start_hv=0.0,
        best_hv=0.0,
        hv_progress=None,
        n_inductions=0,
        n_plans_found=0,
        plan_found_rate=None,
        mean_heldout_accuracy=None,
        mean_prefix_accuracy=None,
        playbook_injection_modes=[],
        wall_s=1.0,
        timed_out=False,
        hit_induction_cap=False,
        hv_progress_measurable=False,
        hv_distinct_values_observed=1,
        hv_exception_count=0,
        hv_sentinel_frac=0.0,
    ).to_row()
    assert row["hv_progress_measurable"] is False
    assert row["hv_progress"] is None, "an unmeasurable cell must not report 0.0 as an observation"
    assert row["hv_distinct_values_observed"] == 1
    assert "hv_sentinel_frac" in row

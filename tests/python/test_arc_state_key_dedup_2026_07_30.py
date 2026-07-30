"""REQ-ARC-WMTE-6051: the in-model search's duplicate-state key is cheap AND partition-preserving.

WHAT CHANGED. `plan_in_model` called `to_ascii(ng)` once per engine call to decide whether it had
already seen a state. A cProfile of a shipped-budget ka59 search attributed 38% of the whole search
to that one function -- ~1.3M `str.join` calls building a one-char-per-cell Python string. It is now
`_state_key`, which does the same job in NumPy.

WHY THESE TESTS EXIST, and why they are not "does it still find a plan".

`to_ascii` renders each cell as `str(int(v))[-1]` -- the LAST DIGIT ONLY -- so it MERGES colour 4
with 14, 5 with 15, 1 with 11, 0 with 10. The shipped dedup key is therefore LOSSY, and those
colours are live in real root grids (ka59 has 4 AND 14 and also 5 AND 15; lp85 has 1/11, 4/14,
5/15; cn04 has 0/10 and 4/14). That has a sharp consequence for this change:

    THE OBVIOUS SWAP IS WRONG. A plain `g.tobytes()` distinguishes those states, so it induces a
    strictly FINER partition -- a change to which states the search explores, not a speedup. It is
    not a theoretical risk: measured across ten games, a plain-bytes key changes the search on
    cn04, taking 93 engine calls / 9 unique states to 140 / 14.

So the property under test is not "the search still works". It is EQUIVALENCE: `_state_key` must
collide on exactly the pairs `to_ascii` collides on, and separate exactly the pairs it separates.
Anything less is a silent change to search behaviour dressed as an optimisation.

Each test below pins one way a plausible "simplification" of `_state_key` would break that
equivalence, and each was checked to FAIL against that simplification rather than merely to pass
against the real thing (7 mutations applied one at a time, 7 killed):

  * drop the `% 10`              -> ``test_the_aliasing_pair_collides_under_both`` (+4 more)
  * drop the shape prefix        -> ``test_shape_is_part_of_the_key``
  * drop the non-negative guard  -> ``test_negative_grids_fall_back_because_mod_10_disagrees_...``
  * drop the integer guard       -> ``test_bool_and_object_dtypes_fall_back_rather_than_crash``
  * drop the `a.size` guard      -> ``test_an_empty_grid_does_not_crash_the_min_guard``
  * fall back unconditionally    -> ``test_the_fast_path_is_actually_taken``
  * uint8 cast before the `% 10` -> ``test_colours_above_255_cannot_wrap_the_uint8_cast``
  * any of the above             -> ``test_equivalence_over_random_grids`` in bulk

TWO OF THOSE TESTS EXIST ONLY BECAUSE AN ADVERSARIAL REVIEW READ THIS FILE AND FOUND IT WANTING,
which is worth recording where the next reader will see it:

  1. The non-negative guard was MISSING, and this docstring, the module docstring, the spec and the
     ops docs all claimed `% 10` matched `to_ascii` "for every integer, negatives included". False:
     `to_ascii` yields the last digit of the ABSOLUTE value (`str(-1)[-1] == "1"`) while
     `-1 % 10 == 9`, so `-1` and `9` -- distinct states -- would have been MERGED. The first eight
     tests all passed against that bug because every one used non-negative colours.
  2. ``test_equivalence_over_random_grids`` carried an anti-vacuity guard that was satisfied by
     exactly the vacuous case it was written to exclude. See its own docstring.
"""

from __future__ import annotations

import numpy as np

from carnot.agentic.arc_executable_world_model import _state_key, plan_in_model, to_ascii

# The colour pairs `to_ascii` cannot tell apart, all of which occur in real ARC root grids.
ALIASING_PAIRS = [(0, 10), (1, 11), (4, 14), (5, 15)]


def _grid(fill: int, shape=(3, 4)) -> np.ndarray:
    return np.full(shape, fill, dtype=np.int16)


def test_the_aliasing_pair_collides_under_both() -> None:
    """The lossy merge is REPRODUCED, not fixed -- that is what makes this a drop-in.

    Dropping the `% 10` would make these grids distinct under `_state_key` while they remain
    identical under `to_ascii`, which is precisely the cn04 search change.
    """
    for low, high in ALIASING_PAIRS:
        a, b = _grid(low), _grid(high)
        assert to_ascii(a) == to_ascii(b), f"premise wrong: to_ascii separates {low}/{high}"
        assert _state_key(a) == _state_key(b), (
            f"_state_key separates {low} from {high} but to_ascii merges them: the partition "
            f"changed, so this is a behaviour change and not an optimisation"
        )


def test_a_plain_bytes_key_would_not_have_been_equivalent() -> None:
    """Pins WHY the `% 10` is required, so a later reader cannot mistake it for ceremony."""
    for low, high in ALIASING_PAIRS:
        a, b = _grid(low), _grid(high)
        assert a.tobytes() != b.tobytes(), (
            f"premise wrong: raw bytes already merge {low}/{high}, so the % 10 would be redundant"
        )
        assert to_ascii(a) == to_ascii(b)
        assert _state_key(a) == _state_key(b)


def test_non_aliasing_colours_stay_distinct() -> None:
    """The cheap key must not over-merge either -- collapsing everything would also 'pass' above."""
    a, b = _grid(4), _grid(5)
    assert to_ascii(a) != to_ascii(b)
    assert _state_key(a) != _state_key(b)


def test_shape_is_part_of_the_key() -> None:
    """`to_ascii` separates rows with newlines, so reshaping changes its output. Raw bytes do not.

    Without the shape prefix these two states would collide. `plan_in_model` happens to be immune
    (it rejects a shape mismatch before keying) but `plan_and_execute` only checks ``ndim == 2``,
    so the prefix is what makes the swap safe at BOTH call sites rather than one.
    """
    flat = np.arange(6, dtype=np.int16)
    a, b = flat.reshape(2, 3), flat.reshape(3, 2)
    assert a.tobytes() == b.tobytes(), "premise wrong: raw bytes already differ, prefix moot"
    assert to_ascii(a) != to_ascii(b)
    assert _state_key(a) != _state_key(b), (
        "a reshaped grid collides with the original: the shape prefix is missing"
    )


def test_negative_float_grid_falls_back_to_to_ascii() -> None:
    """The one input class where `% 10` and `to_ascii` genuinely DISAGREE.

    `to_ascii` truncates toward zero: ``int(-2.7) == -2``, rendering "2". But ``-2.7 % 10 == 7.3``,
    which truncates to 7. Rather than argue that float grids cannot occur, anything that is not an
    integer dtype is handed to `to_ascii` itself, so behaviour is identical by construction.
    """
    neg = np.full((2, 2), -2.7, dtype=np.float64)
    assert _state_key(neg) == to_ascii(neg)
    # And the disagreement it avoids is real, not hypothetical:
    would_have_been = (neg % 10).astype(np.uint8).tobytes()
    assert would_have_been != np.full((2, 2), 2, dtype=np.uint8).tobytes(), (
        "premise wrong: % 10 agrees with to_ascii here, so the fallback guards nothing"
    )


def test_the_fast_path_is_actually_taken() -> None:
    """An unconditional fallback would satisfy every equivalence test above and buy nothing.

    So assert the optimisation is ENGAGED for the input class that matters: a 2-D integer grid must
    come back as `bytes` (the NumPy path), never as the `str` that `to_ascii` returns.
    """
    assert isinstance(_state_key(_grid(7)), bytes)
    assert isinstance(_state_key(np.full((2, 2), 1.5, dtype=np.float32)), str)


def test_equivalence_over_random_grids() -> None:
    """Bulk check: `_state_key` agrees with `to_ascii` on EVERY pair of a corpus built to contain
    BOTH directions -- collides where it collides, separates where it separates.

    THE CORPUS IS CONSTRUCTED, NOT PURELY RANDOM, and that is the whole point. A first version drew
    60 random 4x5 grids from colours 0..15 and guarded against half-vacuity by asserting that the
    corpus contained at least 60 collisions. It did -- and they were EXACTLY the 60 self-pairs.
    Measured over its 1830 pairs: 60 self-pairs, **0 non-trivial collisions**. So the "collides
    where it collides" direction was only ever exercised by comparing a grid with ITSELF, which any
    key whatsoever satisfies, and the guard meant to catch that was satisfied by the very case it
    was written to exclude. Two independently random grids essentially never collide, so collisions
    have to be BUILT.

    Each random grid therefore contributes an aliasing-perturbed TWIN: a distinct array (different
    raw bytes, +10 on a scattered subset of cells) that `to_ascii` cannot tell apart from its
    parent. Those twins are what make the collision direction real.
    """
    rng = np.random.default_rng(6051)
    grids: list[np.ndarray] = []
    for _ in range(30):
        g = rng.integers(0, 6, size=(4, 5), dtype=np.int16)
        grids.append(g)
        twin = g.copy()
        bump = rng.random(g.shape) < 0.5
        twin[bump] += 10  # last digit unchanged -> to_ascii-identical, raw bytes different
        grids.append(twin)

    n_self = n_nontrivial_collision = n_separated = 0
    for i, a in enumerate(grids):
        for j, b in enumerate(grids[i:], start=i):
            ascii_same = to_ascii(a) == to_ascii(b)
            key_same = _state_key(a) == _state_key(b)
            assert ascii_same == key_same, (
                f"disagreement on pair ({i},{j}): to_ascii_same={ascii_same} key_same={key_same}"
            )
            if not ascii_same:
                n_separated += 1
            elif i == j:
                n_self += 1
            else:
                n_nontrivial_collision += 1

    # Both directions must be exercised by NON-TRIVIAL pairs, or the agreement above is half-empty.
    assert n_nontrivial_collision >= 30, (
        f"only {n_nontrivial_collision} non-trivial collisions: the collision direction is vacuous"
    )
    assert n_separated >= 30, (
        f"only {n_separated} separated pairs: the separation direction is thin"
    )
    assert n_self == len(grids)


def test_dtype_variation_does_not_split_a_state() -> None:
    """An LLM-written engine is free to return int64 here and int16 there for the same board.

    A key built on RAW bytes would make those two the same state under one dtype and different
    states under another -- a partition that depends on an engine's incidental typing. `% 10` then
    `uint8` normalises the width away, so the key is value-based exactly as `to_ascii` is.
    """
    values = [[1, 2], [3, 4]]
    variants = [
        np.array(values, dtype=d) for d in (np.int8, np.int16, np.int32, np.int64, np.uint16)
    ]
    keys = {_state_key(v) for v in variants}
    assert len(keys) == 1, f"the same board split into {len(keys)} states by dtype alone"
    assert len({to_ascii(v) for v in variants}) == 1, "premise wrong: to_ascii is dtype-sensitive"


def test_a_non_contiguous_view_keys_the_same_as_its_copy() -> None:
    """Engines slice and transpose. A key that read the memory buffer would treat a view and its
    copy as different states even though they hold the same board, and `to_ascii` -- which iterates
    logically -- would not."""
    base = np.arange(12, dtype=np.int16).reshape(3, 4)
    for view in (np.asfortranarray(base), base.T, base[:, ::2]):
        copy = np.ascontiguousarray(view)
        assert to_ascii(view) == to_ascii(copy), "premise wrong: to_ascii differs on the view"
        assert _state_key(view) == _state_key(copy), (
            "a non-contiguous view keyed differently from its own copy"
        )


def test_colours_above_255_cannot_wrap_the_uint8_cast() -> None:
    """`astype(np.uint8)` would wrap 256 to 0 -- but the `% 10` happens FIRST, so the cast only ever
    sees 0..9 and can never wrap. Pinned because reordering those two operations is an easy and
    entirely silent mistake: it would merge unrelated colours 256 cells apart."""
    for a_val, b_val, should_merge in ((300, 0, True), (300, 302, False), (256, 6, True)):
        a = np.full((2, 2), a_val, dtype=np.int32)
        b = np.full((2, 2), b_val, dtype=np.int32)
        assert (to_ascii(a) == to_ascii(b)) is should_merge
        assert (_state_key(a) == _state_key(b)) is should_merge, (
            f"disagreement on {a_val} vs {b_val}: the uint8 cast may be running before the % 10"
        )


def test_negative_grids_fall_back_because_mod_10_disagrees_with_to_ascii() -> None:
    """THE BUG AN ADVERSARIAL REVIEW CAUGHT BEFORE THIS SHIPPED, pinned so it cannot return.

    The first version of `_state_key` took the arithmetic fast path for ANY integer grid, and its
    docstring, this suite, the spec and the changelog all claimed `% 10` reproduced `to_ascii`
    "EXACTLY for every integer, negatives included". That claim was FALSE. `to_ascii` takes the last
    character of the DECIMAL STRING, which for a negative number is the last digit of its ABSOLUTE
    value -- `str(-1)[-1] == "1"` -- whereas `-1 % 10 == 9`. They agree only where a digit is its own
    complement mod 10, i.e. only for 0 and 5, so they DISAGREE on 12 of the 16 values in -15..-1.

    The original tests all passed because every one of them used non-negative colours, which is the
    "tests test what the author thought to test" mode CLAUDE.md's QA-Layer discipline names.

    The fix declines the fast path for negatives rather than reproducing `to_ascii`'s absolute-value
    semantics in arithmetic (`np.abs(a) % 10` matches everywhere EXCEPT a dtype's most negative
    value, where `abs` silently overflows). So the assertion here is agreement with `to_ascii`
    itself -- by construction, not by argument.
    """
    # Every value where the two encodings disagree must still key exactly as `to_ascii` does.
    for v in range(-15, 0):
        a = np.full((2, 2), v, dtype=np.int16)
        assert _state_key(a) == to_ascii(a), f"negative grid {v} did not fall back to to_ascii"

    # And the disagreement being guarded is real, not defensive ceremony: -1 and -11 are the SAME
    # state under to_ascii, and arithmetic would have agreed here -- but -1 vs 9 is where it breaks.
    neg1 = np.full((2, 2), -1, dtype=np.int16)
    nine = np.full((2, 2), 9, dtype=np.int16)
    assert to_ascii(neg1) != to_ascii(nine), "premise wrong: to_ascii merges -1 with 9"
    assert (neg1 % 10).astype(np.uint8).tobytes() == (nine % 10).astype(np.uint8).tobytes(), (
        "premise wrong: % 10 already separates -1 from 9, so the guard protects nothing"
    )
    assert _state_key(neg1) != _state_key(nine), (
        "-1 and 9 were merged into one state: the fast path is taking negative grids again"
    )

    # A mixed grid (one negative cell) must also fall back -- the guard is on the MINIMUM, not on
    # every cell being negative.
    mixed = np.array([[1, 2], [3, -4]], dtype=np.int16)
    assert _state_key(mixed) == to_ascii(mixed)
    assert isinstance(_state_key(mixed), str)


def test_an_empty_grid_does_not_crash_the_min_guard() -> None:
    """`ndarray.min()` raises on an empty array, so the guard checks `a.size` first. A 0-column grid
    is degenerate but reachable from a misbehaving engine, and keying it must not raise where
    `to_ascii` would simply return an empty-ish string."""
    empty = np.zeros((2, 0), dtype=np.int16)
    assert _state_key(empty) == to_ascii(empty)


def test_bool_and_object_dtypes_fall_back_rather_than_crash() -> None:
    """Neither is an integer dtype, so both must route to `to_ascii` -- whatever it does with them is
    by definition the pre-existing behaviour this change promised not to alter."""
    for arr in (np.array([[True, False]]), np.array([[1, 2]], dtype=object)):
        assert _state_key(arr) == to_ascii(arr)


def test_plan_in_model_finds_the_same_plan_through_the_real_call_sites() -> None:
    """End-to-end through `plan_in_model`, on a grid whose states differ ONLY by an aliasing swap.

    The unit tests above pin the key. This one pins the CALL SITES: a synthetic engine whose only
    reachable states are colour 4 and colour 14 versions of the goal, so the dedup decision is
    load-bearing for which state the search reaches first. It must behave as it did when the key
    was `to_ascii` -- i.e. treat those two as the SAME state.
    """
    start = _grid(0, (2, 2))

    def engine(g, action, data):
        # action 1 -> colour 4 everywhere; action 2 -> colour 14 everywhere (aliases to 4);
        # anything else is a no-op, so the search must dedup rather than loop forever.
        if action == 1:
            return np.full_like(g, 4)
        if action == 2:
            return np.full_like(g, 14)
        return g

    seen_states: list[np.ndarray] = []

    def is_level_complete(g):
        seen_states.append(np.array(g))
        return bool(np.all(g == 14))

    plan = plan_in_model(engine, is_level_complete, start, max_nodes=200, max_depth=6)
    # Under the lossy key the colour-14 state is a DUPLICATE of the colour-4 state, so whichever
    # comes second is discarded and the goal is unreachable. That is the pre-existing behaviour and
    # it must be preserved exactly -- a plain-bytes key would instead find a plan here, which is
    # the search change this REQ refuses to ship.
    assert plan is None, (
        "the colour-14 goal state became reachable, so the dedup key now separates 4 from 14: "
        "the partition changed"
    )
    assert seen_states, "the search never called is_level_complete, so it proved nothing"

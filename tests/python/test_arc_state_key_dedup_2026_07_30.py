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
against the real thing (8 mutations applied one at a time, 8 killed):

  * drop the `% 10`              -> ``test_the_aliasing_pair_collides_under_both`` (+4 more)
  * drop the shape prefix        -> ``test_shape_is_part_of_the_key``
  * drop the non-negative guard  -> ``test_negative_grids_fall_back_because_mod_10_disagrees_...``
  * drop the integer guard       -> ``test_bool_and_object_dtypes_decline_the_arithmetic_rather_than_crash``
  * drop the `a.size` guard      -> ``test_an_empty_grid_does_not_crash_the_min_guard``
  * fall back unconditionally    -> ``test_the_fast_path_is_actually_taken``
  * uint8 cast before the `% 10` -> ``test_colours_above_255_cannot_wrap_the_uint8_cast``
  * split the namespace in two   -> ``test_a_mixed_sign_set_keys_the_same_partition_as_to_ascii``
                                    (+4 more -- reverting to the two-namespace version kills 5)
  * any of the above             -> ``test_equivalence_over_random_grids`` in bulk

THREE OF THOSE TESTS EXIST ONLY BECAUSE AN ADVERSARIAL REVIEW READ THIS FILE AND FOUND IT WANTING,
which is worth recording where the next reader will see it:

  1. The non-negative guard was MISSING, and this docstring, the module docstring, the spec and the
     ops docs all claimed `% 10` matched `to_ascii` "for every integer, negatives included". False:
     `to_ascii` yields the last digit of the ABSOLUTE value (`str(-1)[-1] == "1"`) while
     `-1 % 10 == 9`, so `-1` and `9` -- distinct states -- would have been MERGED. The first eight
     tests all passed against that bug because every one used non-negative colours.
  2. ``test_equivalence_over_random_grids`` carried an anti-vacuity guard that was satisfied by
     exactly the vacuous case it was written to exclude. See its own docstring.
  3. THE FIX FOR (1) WAS ITSELF WRONG, and no amount of additional values would have caught it,
     because every test here asked the WRONG QUESTION. Handing negatives to `to_ascii` verbatim
     satisfies ``_state_key(x) == to_ascii(x)`` per input while BREAKING the partition that is the
     only thing a dedup key is for: a `bytes` never equals a `str`, so ``[[4]]`` and ``[[-4]]`` --
     which `to_ascii` MERGES -- came back SPLIT. Every non-empty 2-D grid now keys into one
     namespace. See ``test_a_mixed_sign_set_keys_the_same_partition_as_to_ascii``, and
     ``_assert_same_partition``, which is how the fallback classes are checked now.
"""

from __future__ import annotations

import numpy as np

from carnot.agentic.arc_executable_world_model import _state_key, plan_in_model, to_ascii

# The colour pairs `to_ascii` cannot tell apart, all of which occur in real ARC root grids.
ALIASING_PAIRS = [(0, 10), (1, 11), (4, 14), (5, 15)]


def _grid(fill: int, shape=(3, 4)) -> np.ndarray:
    return np.full(shape, fill, dtype=np.int16)


def _assert_same_partition(grids: list[np.ndarray], why: str) -> int:
    """Assert the property that actually matters, over a SET: `_state_key` collides on exactly the
    pairs `to_ascii` collides on. Returns the number of pairs compared.

    This helper exists because asserting ``_state_key(x) == to_ascii(x)`` per input -- which is what
    three tests here used to do for the fallback classes -- is a WEAKER and genuinely misleading
    property. See ``test_a_mixed_sign_set_keys_the_same_partition_as_to_ascii``.
    """
    pairs = 0
    for i, a in enumerate(grids):
        for j, b in enumerate(grids[i:], start=i):
            pairs += 1
            ascii_same = to_ascii(a) == to_ascii(b)
            key_same = _state_key(a) == _state_key(b)
            assert ascii_same == key_same, (
                f"{why}: pair ({i},{j}) to_ascii_same={ascii_same} key_same={key_same} "
                f"({to_ascii(a)!r} vs {to_ascii(b)!r}; {_state_key(a)!r} vs {_state_key(b)!r})"
            )
    return pairs


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


def test_negative_float_grid_declines_the_arithmetic_fast_path() -> None:
    """The one input class where `% 10` and `to_ascii` genuinely DISAGREE.

    `to_ascii` truncates toward zero: ``int(-2.7) == -2``, rendering "2". But ``-2.7 % 10 == 7.3``,
    which truncates to 7. So a negative float must not take the arithmetic path -- it takes the
    cell-by-cell path that computes `to_ascii`'s digit directly, in the SAME namespace.
    """
    neg = np.full((2, 2), -2.7, dtype=np.float64)
    # It must key as "2" per cell (to_ascii's digit), NOT as 7 (what the arithmetic would give).
    assert _state_key(neg) == b"2:2|" + bytes([2, 2, 2, 2])
    would_have_been = (neg % 10).astype(np.uint8).tobytes()
    assert would_have_been != bytes([2, 2, 2, 2]), (
        "premise wrong: % 10 agrees with to_ascii here, so the guard protects nothing"
    )
    # And it must land in the same partition as the grid to_ascii merges it with.
    _assert_same_partition(
        [neg, np.full((2, 2), 2, dtype=np.int16), np.full((2, 2), -12.9, dtype=np.float64)],
        "negative float did not share to_ascii's partition",
    )


def test_the_fast_path_is_actually_taken() -> None:
    """An unconditional fallback would satisfy every equivalence test above and buy nothing.

    So assert the optimisation is ENGAGED for the input class that matters: a 2-D non-negative
    integer grid must come back as the arithmetic encoding, which is what carries the measured
    speedup. Checked by VALUE, not by `isinstance`: every non-empty 2-D grid now returns `bytes`
    (that is the partition fix), so a type check no longer distinguishes the paths at all.
    """
    assert _state_key(_grid(7)) == b"3:4|" + bytes([7] * 12)
    # A grid that declines the arithmetic still keys into the same namespace -- only an input
    # `to_ascii` itself collapses (non-2-D, or empty) is handed out verbatim as a `str`.
    assert isinstance(_state_key(np.full((2, 2), 1.5, dtype=np.float32)), bytes)
    assert isinstance(_state_key(np.zeros((2, 0), dtype=np.int16)), str)


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
    # Every value where the two encodings disagree must still key into to_ascii's OWN class: the
    # digit of the absolute value, not `v % 10`.
    for v in range(-15, 0):
        a = np.full((2, 2), v, dtype=np.int16)
        digit = int(str(v)[-1])
        assert _state_key(a) == b"2:2|" + bytes([digit] * 4), (
            f"negative grid {v} keyed as {_state_key(a)!r}, not to_ascii's digit {digit}"
        )

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

    # A mixed grid (one negative cell) must also decline the arithmetic -- the guard is on the
    # MINIMUM, not on every cell being negative.
    mixed = np.array([[1, 2], [3, -4]], dtype=np.int16)
    assert _state_key(mixed) == b"2:2|" + bytes([1, 2, 3, 4])
    _assert_same_partition(
        [mixed, np.array([[1, 2], [3, 4]], dtype=np.int16), np.array([[11, 2], [3, 14]])],
        "a grid with one negative cell left to_ascii's partition",
    )


def test_a_mixed_sign_set_keys_the_same_partition_as_to_ascii() -> None:
    """THE SECOND BUG AN ADVERSARIAL REVIEW CAUGHT, and the one the eight original tests could not
    have caught no matter how many values they tried, because they asserted the WRONG PROPERTY.

    The fix for the negative-colour bug handed negatives to `to_ascii` itself and argued that
    equivalence therefore held "by construction". Every test agreed, because each one asked
    ``_state_key(x) == to_ascii(x)`` -- a PER-INPUT VALUE question. But a dedup key is only ever used
    to compare TWO states, so the property required of it is a PARTITION:

        ``key(x) == key(y)``  iff  ``to_ascii(x) == to_ascii(y)``

    Two namespaces break that while satisfying the per-input reading, because `bytes` never equals
    `str`. `to_ascii` MERGES ``[[4]]`` with ``[[-4]]`` (both render "4"); the two-namespace key
    returned ``b"1:1|\\x04"`` and ``"4"`` and SPLIT them. Any reachable set mixing a negative cell
    with its aliasing twin silently explored states the `to_ascii` key had deduplicated -- the exact
    finer-partition failure the whole change was designed to avoid, reintroduced by its own fix.

    Latent in practice (real ARC grids are non-negative), which is why it is pinned rather than
    merely fixed: engines here are arbitrary LLM-written code, and "the input class does not occur"
    is the assumption that produced this bug twice.
    """
    # The pair that regressed, stated as bluntly as possible.
    pos, neg = np.array([[4]]), np.array([[-4]])
    assert to_ascii(pos) == to_ascii(neg), "premise wrong: to_ascii does not merge 4 with -4"
    assert _state_key(pos) == _state_key(neg), (
        "4 and -4 keyed into different states: the fallback is in a separate namespace again"
    )

    # And in bulk, over a set that deliberately spans every branch: non-negative int (arithmetic),
    # negative int, float, bool, object, and aliasing twins of each.
    corpus = [
        np.array([[4, 5], [0, 1]], dtype=np.int16),  # arithmetic path
        np.array([[14, 15], [10, 11]], dtype=np.int16),  # to_ascii-identical twin
        np.array([[-4, 5], [0, 1]], dtype=np.int16),  # one negative -> cell-by-cell path
        np.array([[-14, -5], [0, -11]], dtype=np.int16),  # all negative, same digits
        np.array([[4.0, 5.2], [0.9, 1.1]], dtype=np.float64),  # float truncation
        np.array([[-4.7, 5.0], [0.0, 1.0]], dtype=np.float64),
        np.array([[4, 5], [0, 1]], dtype=object),
        np.array([[True, False], [False, True]]),
        np.array([[1, 0], [0, 1]], dtype=np.int16),  # what bool must merge with
        np.array([[9, 9], [9, 9]], dtype=np.int16),  # must NOT merge with -1 grids
        np.array([[-1, -1], [-1, -1]], dtype=np.int16),
        np.array([[-11, -21], [-31, -41]], dtype=np.int16),  # same digits as the -1 grid
    ]
    pairs = _assert_same_partition(corpus, "mixed-branch corpus left to_ascii's partition")
    # Anti-vacuity: the set must actually contain merges AND separations across DIFFERENT branches,
    # or the agreement above is trivially satisfied by any key at all.
    merged = sum(
        1 for i, a in enumerate(corpus) for b in corpus[i + 1 :] if to_ascii(a) == to_ascii(b)
    )
    assert pairs == 78
    assert merged >= 6, f"only {merged} non-trivial merges: the collision direction is vacuous"


def test_an_empty_grid_does_not_crash_the_min_guard() -> None:
    """`ndarray.min()` raises on an empty array, so the guard checks `a.size` first. A 0-column grid
    is degenerate but reachable from a misbehaving engine, and keying it must not raise where
    `to_ascii` would simply return an empty-ish string."""
    empty = np.zeros((2, 0), dtype=np.int16)
    assert _state_key(empty) == to_ascii(empty)


def test_bool_and_object_dtypes_decline_the_arithmetic_rather_than_crash() -> None:
    """Neither is an integer dtype, so both decline the arithmetic and take the cell-by-cell path --
    which must reproduce `to_ascii`'s digit, and must land bool in the same class as the 0/1 integer
    grid `to_ascii` cannot tell it apart from."""
    for arr in (np.array([[True, False]]), np.array([[1, 0]], dtype=object)):
        assert _state_key(arr) == b"1:2|" + bytes([1, 0]), f"{arr.dtype} keyed unexpectedly"
    _assert_same_partition(
        [
            np.array([[True, False]]),
            np.array([[1, 0]], dtype=object),
            np.array([[1, 0]], dtype=np.int16),
            np.array([[11, 10]], dtype=np.int16),
            np.array([[1, 1]], dtype=np.int16),
        ],
        "bool/object left to_ascii's partition",
    )


def test_the_dtype_guard_is_checked_before_min_and_not_merely_alongside_it() -> None:
    """The `dtype.kind in "iu"` test must SHORT-CIRCUIT `a.min()`, not just narrow the fast path.

    ADDED 2026-07-30, by re-running this file's mutation proof under `-n0`. The recorded claim was
    "mutation-proven 7/7" with no method stated; re-run it is **7 of 8**, and this is the survivor.

    The surviving mutant relaxes `a.dtype.kind in "iu" and a.min() >= 0` to `a.min() >= 0` alone.
    On every grid this repo's own tests exercise that is INDISTINGUISHABLE -- for a non-negative
    float, `int(v) % 10 == int(v % 10)`, verified over 400k random values and every large/precision
    edge case in float32 and float64, so the integer check looks redundant once the sign check is
    present. It is not redundant, for a reason none of those tests reach: `a.min()` is only DEFINED
    for an orderable dtype. On a string grid HEAD never evaluates it (the `kind` check is False and
    `and` short-circuits) and falls through to the per-cell path, which keys it correctly. The
    mutant evaluates it and dies with `UFuncTypeError`.

    Why a string grid is worth pinning rather than dismissing: the function's own docstring already
    declines to assume engine output is well-behaved -- "engines here are arbitrary LLM-written
    code, which is precisely the class of assumption that should not be load-bearing" -- and that
    argument does not distinguish a negative cell from a stringly-typed one. An engine that builds
    its grid with `np.array([[str(v) ...]])` is a plausible LLM bug, and the difference between
    "keys correctly" and "raises inside the dedup path" is the difference between a search that
    works and a round that dies.
    """
    # '4'/'14' and '5'/'15' are the aliasing pairs, so this grid also exercises the merge rather
    # than merely surviving the call.
    stringly = np.array([["4", "14"], ["5", "15"]])
    swapped = np.array([["14", "4"], ["15", "5"]])
    collapsed = np.array([["4", "4"], ["5", "5"]])

    key = _state_key(stringly)
    assert isinstance(key, bytes), (
        "a string grid must key into the same bytes namespace as every other non-empty 2-D grid; "
        "a str key here would re-open the two-namespace partition break this file exists to pin"
    )
    _assert_same_partition(
        [stringly, swapped, collapsed],
        "a string-dtype grid keys by to_ascii's last digit like any other",
    )

    # The guard order is the actual property. Stated as the mutant's own failure: evaluating
    # `min()` on this dtype is what HEAD must avoid.
    try:
        _ = stringly.min() >= 0
    except Exception:
        pass
    else:  # pragma: no cover - defensive; numpy raising here is what makes the guard load-bearing
        raise AssertionError(
            "numpy now orders this dtype, so the short-circuit is no longer observable and this "
            "test needs a dtype that still raises -- it must not be left silently vacuous"
        )


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

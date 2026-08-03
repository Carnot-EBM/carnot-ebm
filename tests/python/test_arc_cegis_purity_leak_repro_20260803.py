"""CHARACTERIZATION tests for the CEGIS acceptance/refinement purity leak (defect C).

These tests assert the CURRENT, DEFECTIVE behaviour on purpose. They are the target the fix
phase has to move: whoever ships `CARNOT_ARC_CEGIS_ACCEPT_SPLIT` must make the flag-OFF path
still satisfy every assertion here (that is what "OFF reproduces shipped behaviour exactly,
including the bug" means) and add the flag-ON counterparts alongside them.

WHY CHARACTERIZE RATHER THAN JUST DESCRIBE. The defect was reported once, withdrawn, and
re-reported. A test is the only form of a claim that cannot quietly stop being true: if some
future edit to `_bounded_mismatches`, `WorldModelVerifier.score`'s `max_mismatch`, or
`_split_prefix_heldout` changes the leak, this goes red and someone has to look.

Spec: REQ-ARC-WMTE-4544 (the re-induction gate that attaches real mismatch evidence to the
refactor call), REQ-ARC-WMTE-4791 (`select_trusted_world_model`, which owns the prefix/held-out
split acceptance is scored on), REQ-ARC-FCP-5699-26 (`_bounded_mismatches`, the five-mismatch
render cap). No NEW REQ is shipped by this reproduce-only phase; these three are cited because
the defect is their INTERACTION -- each is individually as specified, and the leak is what
happens when the rows 4791 grades on are the rows 4544 feeds back through 5699-26's cap.

Artifact: results/outer_loop_arc_cegis_purity_leak_20260803.json
"""

from __future__ import annotations

import json

import numpy as np

from carnot.agentic.arc_executable_world_model import (
    Transition,
    WorldModelVerifier,
    _bounded_mismatches,
    _delta,
    refactor_prompt,
)
from carnot.agentic.arc_llm_reinduction import _counterexample_result
from carnot.agentic.arc_world_model_trust_energy import (
    WorldModelCandidate,
    _split_prefix_heldout,
    select_trusted_world_model,
)


def _corpus(n: int, *, terminal_levelup: bool = True) -> list[Transition]:
    """Each step changes exactly one, distinct cell -- so a rendered delta identifies its row."""
    g = np.zeros((8, 8), dtype=int)
    rows: list[Transition] = []
    for i in range(n):
        nxt = g.copy()
        nxt[i // 8, i % 8] = i + 3
        rows.append(
            Transition(
                g.copy(), i % 4, None, nxt.copy(), 0, 1 if (terminal_levelup and i == n - 1) else 0
            )
        )
        g = nxt
    return rows


def _engine(rows: list[Transition], correct: set[int]):
    table = {
        (rows[i].grid.tobytes(), int(rows[i].action)): rows[i].next_grid.copy()
        for i in sorted(correct)
    }

    def engine(grid, action, data=None):
        hit = table.get((np.asarray(grid).tobytes(), int(action)))
        return hit.copy() if hit is not None else np.asarray(grid).copy()

    return engine


def _delivered_rows(rows: list[Transition], prompt: str) -> set[int]:
    """Rows whose OBSERVED answer is present in the rendered prompt TEXT.

    Parsed out of the prompt rather than read off the mismatch dicts: availability is not
    delivery, and rows are identified by their own unique changed cell, never by the
    mismatch's `i` label.
    """
    start = prompt.find("MISMATCHES:\n")
    assert start >= 0, "refactor_prompt no longer emits a MISMATCHES block"
    body = prompt[start + len("MISMATCHES:\n") :]
    end = body.rfind("]")
    parsed = json.loads(body[: end + 1])
    tup2row = {}
    for i, t in enumerate(rows):
        for tup in _delta(t.grid, t.next_grid):
            tup2row.setdefault(tuple(tup), i)
    out: set[int] = set()
    for m in parsed:
        for tup in m.get("true_change") or []:
            hit = tup2row.get(tuple(tup))
            if hit is not None:
                out.add(hit)
    return out


def _run(rows: list[Transition], correct: set[int]):
    prefix, heldout = _split_prefix_heldout(rows)
    eng = _engine(rows, correct)
    sel = select_trusted_world_model(
        list(rows), [WorldModelCandidate("c", eng, None)], hidden_state=True
    )
    vr = WorldModelVerifier(list(rows), hud_mask=None).score(eng)
    prompt = refactor_prompt(
        "SYNTH",
        _counterexample_result(
            {
                "real_n": vr.n,
                "real_n_correct": vr.n_correct,
                "real_accuracy": float(vr.accuracy),
                "real_mismatches": list(vr.mismatches),
            }
        ),
    )
    return prefix, heldout, sel, vr, prompt


def test_prefix_perfect_engine_leaks_every_gradeable_heldout_answer_at_n12():
    """The regime that matters: prefix-perfect, held-out failing.

    At the measured offline-harness shape (n=12 -> prefix 8, held-out 4, of which 3 are
    gradeable because the terminal level-up row is excluded from scoring), EVERY gradeable
    acceptance row's observed answer is delivered into the refinement prompt.
    """
    rows = _corpus(12)
    prefix, heldout, sel, vr, prompt = _run(rows, set(range(len(_split_prefix_heldout(rows)[0]))))
    assert len(prefix) == 8 and len(heldout) == 4
    gradeable = [
        i for i in range(len(prefix), len(rows)) if rows[i].level_after <= rows[i].level_before
    ]
    assert gradeable == [8, 9, 10]
    assert float(sel.selected_score.prefix_accuracy) == 1.0
    assert float(sel.selected_score.heldout_accuracy) == 0.0  # gate would REJECT this round
    assert set(gradeable).issubset(_delivered_rows(rows, prompt))


def test_leak_is_monotone_in_prefix_quality_and_absent_for_a_floor_engine():
    """A bad-on-prefix engine crowds the render budget with prefix mismatches; a good one does
    not. This is the "worst precisely when the engine is good" claim, as an assertion."""
    rows = _corpus(25)
    n_prefix = len(_split_prefix_heldout(rows)[0])
    assert n_prefix == 17
    counts = []
    for p in (0, 8, 13, 15, 17):
        _pre, _hel, _sel, _vr, prompt = _run(rows, set(range(p)))
        counts.append(len({i for i in _delivered_rows(rows, prompt) if i >= n_prefix}))
    assert counts[0] == 0, "a floor engine should not leak at n=25"
    assert counts == sorted(counts), f"leak must be monotone in prefix quality, got {counts}"
    assert counts[-1] == 5, "a prefix-perfect engine should fill every render slot from the tail"


def test_leak_law_is_exact():
    """n_leaked = max(0, min(5 - n_prefix_mismatches, n_heldout_mismatches)).

    Deleting either bound below makes this fail, which is how each half is proven to bite:
    the `5` is `_bounded_mismatches`'s render limit, the index ordering is
    `WorldModelVerifier.score`'s append order.
    """
    for n in (12, 25):
        rows = _corpus(n)
        n_prefix = len(_split_prefix_heldout(rows)[0])
        for p in range(n_prefix + 1):
            _pre, _hel, _sel, vr, prompt = _run(rows, set(range(p)))
            idx = [int(m["i"]) for m in vr.mismatches if "i" in m]
            n_pre_mis = sum(1 for i in idx if i < n_prefix)
            n_hel_mis = sum(1 for i in idx if i >= n_prefix)
            expected = max(0, min(5 - n_pre_mis, n_hel_mis))
            actual = len({i for i in _delivered_rows(rows, prompt) if i >= n_prefix})
            assert actual == expected, f"n={n} p={p}: expected {expected}, got {actual}"


def test_prefix_only_refinement_is_starved_exactly_where_the_leak_is_worst():
    """The measured cost of design (i). At prefix-perfect there are ZERO prefix mismatches, so
    prefix-only refinement would hand the LLM no counterexample at all -- in the same round
    where the full-corpus path hands it the entire acceptance set."""
    for n in (12, 25):
        rows = _corpus(n)
        prefix, _heldout = _split_prefix_heldout(rows)
        eng = _engine(rows, set(range(len(prefix))))
        assert WorldModelVerifier(list(prefix), hud_mask=None).score(eng).mismatches == []
        full = WorldModelVerifier(list(rows), hud_mask=None).score(eng).mismatches
        assert full, "the full-corpus path still finds counterexamples in the same round"


def test_render_cap_and_collection_cap_are_the_shipped_values():
    """The leak law depends on two literals. Pin them, so a silent change to either shows up
    here rather than as a mysterious change in gate behaviour months later."""
    rows = _corpus(25)
    vr = WorldModelVerifier(list(rows), hud_mask=None).score(_engine(rows, set()))
    assert len(vr.mismatches) == 8, "WorldModelVerifier.score max_mismatch default moved"
    assert len(_bounded_mismatches(list(vr.mismatches))) == 5, "_bounded_mismatches limit moved"

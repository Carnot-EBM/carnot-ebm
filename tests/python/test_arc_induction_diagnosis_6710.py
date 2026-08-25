"""REQ-ARC-WMTE-6710 / SCENARIO-ARC-WMTE-6710-1..7.

WHAT THESE TESTS PROTECT, AND WHY THEY EXIST.

`induction_skipped` is the field this project uses to answer "why was the induced world model
rejected". Before this REQ it could not resolve that question, in three separate ways.

  (A) `execute_bounded_llm_reinduction` pre-sets `skipped =
      "no_reachable_plan_after_refinement"` before any round runs. Two sites diagnosed a real
      cause and wrote it ONLY to the per-round record -- the held-out dynamics failure, and the
      selection/planning exception. So an attempt whose every round failed DYNAMICS verification
      was reported under a PLANNING label, and a reader chasing it went to the planner instead of
      to the dynamics model. That default is 9 of the 120 skip records the corpus holds
      outside the induction-disabled case (measured 2026-08-24 over `results/**`).

  (B) The per-round records reached the harness and were read only to build category counters.
      The evidence naming WHICH cause fired was rebuilt every cell and discarded every cell.

  (C) The proposer folded the model's thought and answer channels into three `last_*` fields
      that hold only the most recent call, so every earlier completion's split died in memory.
      "The model spends its budget thinking and returns nothing" stayed an inference.

The regression test named for the incident input is
`test_all_rounds_failing_heldout_verification_is_not_reported_as_a_planning_failure`.

EVIDENCE SAFETY: every test that touches the engine store redirects it to `tmp_path`. Nothing
here can write `results/arc_e3`, which is measurement evidence and must never be mutated by a
test run.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

from carnot.agentic import arc_executable_world_model as e3
from carnot.agentic import arc_llm_reinduction as _reinduction
from carnot.agentic.arc_executable_world_model import Transition
from carnot.agentic.arc_llm_reinduction import (
    _channel_totals_delta,
    _channel_totals_snapshot,
    execute_bounded_llm_reinduction,
)

GAME = "diag"
_MOVES = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
# Fixed, not sampled. The held-out tail must contain an action-3 tick, so an engine that models
# only the move scores BELOW the 1.0 dynamics veto -- which is what puts the loop on the
# held-out-failure path this REQ is about.
ACTIONS = [1, 3, 0, 2, 1, 3, 0, 2, 3]

_HARNESS = Path(__file__).resolve().parents[2] / "scripts" / "arc_scored_path_lever_harness.py"


@pytest.fixture(autouse=True)
def _multi_round_cap(monkeypatch):
    """MAX_REFINEMENT_ROUNDS is capped at 1 on the shipped path. These tests need several rounds
    so "EVERY round failed held-out verification" is a real sequence rather than one round. The
    shipped cap is pinned separately by tests/python/test_arc_refinement_rounds_cap.py."""

    monkeypatch.setattr(_reinduction, "MAX_REFINEMENT_ROUNDS", 3)


def _true_next(grid: np.ndarray, action: int) -> np.ndarray:
    g = grid.copy()
    pos = np.argwhere(g == 3)
    r, c = int(pos[0][0]), int(pos[0][1])
    dr, dc = _MOVES[int(action) % 4]
    g[r, c] = 0
    g[(r + dr) % g.shape[0], (c + dc) % g.shape[1]] = 3
    if int(action) % 4 == 3:
        g[5, 5] = int(g[5, 5]) + 1
    return g


def _corpus() -> tuple[list[Transition], np.ndarray]:
    grid = np.zeros((6, 6), dtype=int)
    grid[2, 2] = 3
    root = grid.copy()
    rows: list[Transition] = []
    for action in ACTIONS:
        nxt = _true_next(grid, action)
        rows.append(
            Transition(
                grid=grid.copy(),
                action=action,
                data=None,
                next_grid=nxt.copy(),
                level_before=0,
                level_after=0,
            )
        )
        grid = nxt
    return rows, root


# Models the move, NOT the action-3 counter tick: held-out accuracy 0.6667, so it fails the 1.0
# dynamics veto on every round.
PARTIAL_SRC = """
import numpy as np

_MOVES = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}


def engine(grid, action, data):
    g = np.asarray(grid).copy()
    pos = np.argwhere(g == 3)
    if len(pos) == 0:
        return g
    r, c = int(pos[0][0]), int(pos[0][1])
    dr, dc = _MOVES.get(int(action) % 4, (0, 0))
    g[r, c] = 0
    g[(r + dr) % g.shape[0], (c + dc) % g.shape[1]] = 3
    return g


def is_level_complete(grid):
    return False
"""

# Models the tick too, so held-out accuracy is 1.0 and the dynamics veto PASSES. Its goal is
# SATISFIABLE and reachable (the player at the origin of a torus), so the loop gets past the
# degenerate-goal veto as well -- and then the stub planner returns nothing. That combination is
# the genuine no-plan case. A constant-False goal would NOT do: it reports
# `degenerate_goal_predicate`, a different cause, which is what the first draft of this test got
# wrong and what this test now pins.
PERFECT_SRC = """
import numpy as np

_MOVES = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}


def engine(grid, action, data):
    g = np.asarray(grid).copy()
    pos = np.argwhere(g == 3)
    if len(pos) == 0:
        return g
    r, c = int(pos[0][0]), int(pos[0][1])
    dr, dc = _MOVES.get(int(action) % 4, (0, 0))
    g[r, c] = 0
    g[(r + dr) % g.shape[0], (c + dc) % g.shape[1]] = 3
    if int(action) % 4 == 3:
        g[5, 5] = int(g[5, 5]) + 1
    return g


def is_level_complete(grid):
    g = np.asarray(grid)
    pos = np.argwhere(g == 3)
    return bool(len(pos) and int(pos[0][0]) == 0 and int(pos[0][1]) == 0)
"""


class _ScriptedProposer:
    """Replays one engine source per round, writing each into the store, exactly as the real
    proposer does. Nothing in this REQ may depend on the proposer being an LLM."""

    model_specs = "scripted-diagnosis-test-proposer"

    def __init__(self, store: Path, game: str, sources: list[str]) -> None:
        self.store = Path(store)
        self.game = game
        self.sources = list(sources)
        self.writes: list[int] = []

    def _write(self, index: int) -> tuple[bool, str]:
        src = self.sources[min(index, len(self.sources) - 1)]
        path = self.store / self.game / "world_model.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(src)
        self.writes.append(index)
        return True, f"wrote round {index + 1}"

    def induce(self, game, trans, cell, *, previous_level_complete_grid=None):
        return self._write(0)

    def refactor(self, game, vr):
        return self._write(len(self.writes))


def _run(monkeypatch, tmp_path, sources, *, load_engine=None, plan_in_model=None):
    """Drive the REAL `execute_bounded_llm_reinduction`. The selection and verification path is
    not stubbed: the claim under test is about what that real path reports."""

    monkeypatch.setattr(e3, "E3_DIR", tmp_path)
    transitions, root = _corpus()
    proposer = _ScriptedProposer(tmp_path, GAME, sources)
    return execute_bounded_llm_reinduction(
        game=GAME,
        transitions=transitions,
        cell=1,
        root_grid=root,
        proposer=proposer,
        candidate_provider=lambda engine, goal: [("loaded_world_model.py", engine, goal)],
        load_engine=load_engine or e3.load_engine,
        plan_in_model=plan_in_model or (lambda engine, goal, grid: None),
        max_rounds=3,
        min_heldout_accuracy=1.0,
    )


def _harness() -> ModuleType:
    """Load the lever harness by path; it lives in scripts/, not an installed package."""

    spec = importlib.util.spec_from_file_location("arc_scored_path_lever_harness", _HARNESS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# =============================================================================================
# DEFECT (A): a diagnosed cause must leave the round that diagnosed it.
# =============================================================================================


def test_all_rounds_failing_heldout_verification_is_not_reported_as_a_planning_failure(
    monkeypatch, tmp_path
):
    """SCENARIO-ARC-WMTE-6710-1. THE REGRESSION TEST NAMED FOR THE INCIDENT INPUT.

    Every round fails held-out transition verification, which is a DYNAMICS failure. Before the
    fix this attempt reported `no_reachable_plan_after_refinement` -- a PLANNING label -- because
    the site wrote its diagnosis only into the per-round record. 9 of the 120 non-disabled skip
    records in the corpus were that default.

    Mutation M1: delete `skipped = row["skipped"]` at the held-out site. RED here.
    """

    outcome = _run(monkeypatch, tmp_path, [PARTIAL_SRC, PARTIAL_SRC, PARTIAL_SRC])

    assert outcome.skipped == "heldout_transition_verification_failed", (
        f"a dynamics failure must be reported as a dynamics failure; got {outcome.skipped!r}"
    )
    assert outcome.skipped != "no_reachable_plan_after_refinement"
    assert [r.get("skipped") for r in outcome.rounds] == [
        "heldout_transition_verification_failed"
    ] * 3


def test_a_raising_round_is_not_reported_as_a_planning_failure(monkeypatch, tmp_path):
    """SCENARIO-ARC-WMTE-6710-2. The exception site had the same defect: an attempt that RAISED
    was reported as an attempt that found no plan, sending a reader to the planner instead of to
    the traceback.

    Mutation M2: delete `skipped = row["skipped"]` at the exception site. RED here.
    """

    def _boom(game):
        raise ValueError("engine load exploded")

    outcome = _run(monkeypatch, tmp_path, [PARTIAL_SRC], load_engine=_boom)

    assert outcome.skipped == "selection_or_planning_exception", (
        f"a raising round must report the exception; got {outcome.skipped!r}"
    )
    assert outcome.skipped != "no_reachable_plan_after_refinement"


def test_a_genuine_no_plan_attempt_still_reports_the_no_plan_default(monkeypatch, tmp_path):
    """SCENARIO-ARC-WMTE-6710-3. The fix must NARROW the default, not delete it. An attempt whose
    dynamics verify cleanly and whose planner finds nothing is genuinely a planning failure and
    must still say so -- otherwise the widening would trade one wrong label for another."""

    outcome = _run(monkeypatch, tmp_path, [PERFECT_SRC, PERFECT_SRC, PERFECT_SRC])

    assert outcome.skipped == "no_reachable_plan_after_refinement", (
        f"a real no-plan attempt must keep the no-plan label; got {outcome.skipped!r}"
    )
    assert all(
        r.get("skipped") != "heldout_transition_verification_failed" for r in outcome.rounds
    ), "the dynamics veto must have PASSED for this to be a planning test"


def test_the_round_record_carries_its_own_heldout_score_and_verdict(monkeypatch, tmp_path):
    """SCENARIO-ARC-WMTE-6710-1 (record half). The harness digest reads `heldout_accuracy` and
    `accepted_by_heldout_verifier` off the ROW. Both are written by the round's `row.update(...)`
    block, and nothing pinned that contract, so a future edit to that block could silently make
    the digest publish None for every round without any test noticing.

    Mutation M3: drop `"heldout_accuracy"` from the round's `row.update(...)`. RED here.
    Mutation M4: drop `"accepted_by_heldout_verifier"` from the same block. RED here.
    """

    outcome = _run(monkeypatch, tmp_path, [PARTIAL_SRC, PARTIAL_SRC, PARTIAL_SRC])

    for row in outcome.rounds:
        assert isinstance(row.get("heldout_accuracy"), float), (
            f"round {row.get('round')} carries no held-out score: {sorted(row)}"
        )
        assert row["heldout_accuracy"] == pytest.approx(2 / 3, abs=1e-3)
        assert row.get("accepted_by_heldout_verifier") is False


# =============================================================================================
# DEFECT (B): the per-round record must survive into the harness row.
# =============================================================================================


def test_the_harness_keeps_a_bounded_per_round_record():
    """SCENARIO-ARC-WMTE-6710-4. The harness read the rounds only to count categories, so the
    evidence behind a count was discarded every cell.

    Mutation M5: make `_round_digest` return []. RED here.
    """

    m = _harness()
    atts = [
        {
            "reason": "stagnation",
            "skipped": "heldout_transition_verification_failed",
            "refinement_rounds": [
                {
                    "round": i + 1,
                    "action": "induce" if i == 0 else "refactor",
                    "skipped": "heldout_transition_verification_failed",
                    "heldout_accuracy": 0.667,
                    "accepted_by_heldout_verifier": False,
                    "plan_reaches_goal": False,
                    "counterexample": {
                        "kind": "heldout_transition_mismatch",
                        "real_accuracy": 0.667,
                        "real_mismatches": [{"cell": j} for j in range(40)],
                    },
                }
                for i in range(9)
            ],
            "counterexamples": [{"kind": "heldout_transition_mismatch", "mismatches": [1, 2, 3]}],
        }
    ]

    records = m._induction_round_records(atts)

    assert len(records) == 1
    entry = records[0]
    assert entry["attempt"] == 0
    assert entry["skipped"] == "heldout_transition_verification_failed"
    assert len(entry["rounds"]) == m._ROUND_DIGEST_MAX_ROUNDS, (
        "rounds must be bounded -- these are written per cell"
    )
    first = entry["rounds"][0]
    assert first["heldout_accuracy"] == 0.667
    assert first["accepted_by_heldout_verifier"] is False
    # Every list becomes a length under a `<key>_n` name, so a reader can see it was summarised
    # rather than dropped. The grids themselves are the largest thing on a round.
    assert first["counterexample"]["real_mismatches_n"] == 40
    assert "real_mismatches" not in first["counterexample"]
    assert entry["counterexamples"][0]["mismatches_n"] == 3


# =============================================================================================
# DEFECT (C): the channel split must be counted, not overwritten.
# =============================================================================================


def _proposer_with_counters():
    """A bare LocalGGUFProposer instance, built without touching a GPU or a server. Only the
    counter fields and the two accumulation seams are exercised."""

    return e3.LocalGGUFProposer.__new__(e3.LocalGGUFProposer)


def test_the_channel_split_is_accumulated_across_completions():
    """SCENARIO-ARC-WMTE-6710-5. The three `last_*` fields hold only the most recent call, so
    every earlier completion's split died in memory.

    Mutation M6: delete the three `channel_totals[...] += ...` lines at the chat split seam.
    RED here.
    """

    p = _proposer_with_counters()
    p.channel_totals = dict(e3._EMPTY_CHANNEL_TOTALS)
    p.last_final_content = ""
    p.last_reasoning_content = ""

    for final, reasoning in (("abcd", "x" * 100), ("ef", "y" * 50)):
        e3.LocalGGUFProposer._record_chat_channel_split(p, final, reasoning)

    assert p.channel_totals["chat_completions"] == 2
    assert p.channel_totals["chars_final"] == 6
    assert p.channel_totals["chars_reasoning"] == 150
    assert p.channel_totals["reasoning_only"] == 0
    assert p.channel_totals["both_channels_empty"] == 0


def test_an_empty_answer_channel_is_counted_as_its_own_shape():
    """SCENARIO-ARC-WMTE-6710-6. This is the shape the whole REQ is for: a call that spends its
    entire budget on the thought channel and returns no code. It must be distinguishable from a
    call that returned nothing at all, and from an ordinary short answer."""

    p = _proposer_with_counters()
    p.channel_totals = dict(e3._EMPTY_CHANNEL_TOTALS)
    p.last_final_content = ""
    p.last_reasoning_content = ""

    e3.LocalGGUFProposer._record_chat_channel_split(p, "   ", "thought " * 500)
    e3.LocalGGUFProposer._record_chat_channel_split(p, "", "")

    assert p.channel_totals["reasoning_only"] == 1
    assert p.channel_totals["both_channels_empty"] == 1
    assert p.channel_totals["chat_completions"] == 2


def test_a_round_publishes_its_own_numbers_not_a_run_total():
    """SCENARIO-ARC-WMTE-6710-7. The counters are cumulative for the whole process, so a round
    that published them raw would report every earlier round's characters too. Differencing two
    reads is what makes them a per-round measurement.

    Mutation M7: make `_channel_totals_delta` return `dict(after)`. RED here.
    """

    before = {"completions": 3, "chars_final": 900, "chars_reasoning": 60000}
    after = {"completions": 4, "chars_final": 1000, "chars_reasoning": 95000}

    assert _channel_totals_delta(before, after) == {
        "completions": 1,
        "chars_final": 100,
        "chars_reasoning": 35000,
    }
    # A reset between the two reads is not a measurement; it must clamp rather than publish a
    # negative count.
    assert _channel_totals_delta({"chars_final": 500}, {"chars_final": 0}) == {"chars_final": 0}


def test_the_snapshot_is_inert_on_a_proposer_without_counters():
    """A stub proposer -- every scripted proposer in this test suite, and any older proposer
    class -- must not raise or change behaviour. This is observation only."""

    assert _channel_totals_snapshot(object()) == {}
    assert _channel_totals_snapshot(None) == {}
    assert _channel_totals_delta({}, {}) == {}


def test_every_round_carries_a_channel_record_even_when_it_failed(monkeypatch, tmp_path):
    """A round that produced no usable code still spent a budget, and that is exactly the case
    worth counting. The field must be present on every round, not only successful ones."""

    outcome = _run(monkeypatch, tmp_path, [PARTIAL_SRC, PARTIAL_SRC, PARTIAL_SRC])

    assert outcome.rounds, "the loop must have run at least one round"
    for row in outcome.rounds:
        assert "channel_chars" in row, (
            f"round {row.get('round')} carries no channel record: {sorted(row)}"
        )
        assert isinstance(row["channel_chars"], dict)


# =============================================================================================
# FIXES FROM THE ADVERSARIAL REVIEW OF COMMIT 288ea485f9. Each test below exists because a
# hostile reviewer found the first version of this REQ wrong in a specific, reproducible way.
# =============================================================================================


class _CountingProposer(_ScriptedProposer):
    """A scripted proposer that ALSO carries the real per-channel counters.

    The plain `_ScriptedProposer` has no `channel_totals`, so every integration test above saw
    `channel_chars == {}` and could not tell a per-round delta from a run total. That is what let
    the loop-side `channels_before` read be decorative.
    """

    def __init__(self, store, game, sources, per_round_chars):
        super().__init__(store, game, sources)
        self.channel_totals = dict(e3._EMPTY_CHANNEL_TOTALS)
        self.per_round_chars = list(per_round_chars)

    def _write(self, index):
        # Simulate one completion for this round, accumulating the way the real proposer does.
        chars = self.per_round_chars[min(index, len(self.per_round_chars) - 1)]
        self.channel_totals["completions"] += 1
        self.channel_totals["chars_raw"] += chars
        return super()._write(index)


def test_a_round_reports_its_own_characters_not_every_earlier_round_s(monkeypatch, tmp_path):
    """The loop must read the counters BEFORE the round's request. Nothing pinned that: a hostile
    reviewer replaced `channels_before` with `{}` -- which makes every round publish the whole-run
    cumulative total, exactly the failure this REQ is about -- and the suite stayed GREEN.

    Mutation M8: `channels_before = {}` in the round loop. RED here.
    """

    monkeypatch.setattr(e3, "E3_DIR", tmp_path)
    transitions, root = _corpus()
    proposer = _CountingProposer(tmp_path, GAME, [PARTIAL_SRC] * 3, [100, 200, 400])
    outcome = execute_bounded_llm_reinduction(
        game=GAME,
        transitions=transitions,
        cell=1,
        root_grid=root,
        proposer=proposer,
        candidate_provider=lambda engine, goal: [("loaded_world_model.py", engine, goal)],
        load_engine=e3.load_engine,
        plan_in_model=lambda engine, goal, grid: None,
        max_rounds=3,
        min_heldout_accuracy=1.0,
    )

    per_round = [r["channel_chars"].get("chars_raw") for r in outcome.rounds]
    assert per_round == [100, 200, 400], (
        f"each round must report its OWN characters; got {per_round}. "
        "A running total would read [100, 300, 700]."
    )
    assert [r["channel_chars"].get("completions") for r in outcome.rounds] == [1, 1, 1]


def test_an_earlier_dynamics_failure_does_not_outrank_a_later_no_plan_ending(monkeypatch, tmp_path):
    """Round 1 fails held-out verification; rounds 2 and 3 VERIFY and fail only at planning. The
    attempt therefore ended on the planner, and must say so.

    The first version of this REQ reported `heldout_transition_verification_failed` here, because
    the no-plan fall-through was the one loop exit that never assigned `skipped` -- so the fix for
    a wrong label introduced a wrong label in the other direction. Latent on the shipped path
    (MAX_REFINEMENT_ROUNDS defaults to 1) and live whenever the cap is raised.

    Mutation M9: delete `skipped = "no_reachable_plan_after_refinement"` at the loop tail. RED.
    """

    outcome = _run(monkeypatch, tmp_path, [PARTIAL_SRC, PERFECT_SRC, PERFECT_SRC])

    per_round = [r.get("skipped") for r in outcome.rounds]
    assert per_round[0] == "heldout_transition_verification_failed", (
        f"round 1 must still record its own dynamics failure; got {per_round}"
    )
    assert per_round[1:] == [None, None], (
        f"rounds 2-3 must have verified and fallen through to the planner; got {per_round}"
    )
    assert outcome.skipped == "no_reachable_plan_after_refinement", (
        f"the attempt ended on the planner, so it must report the planner; got {outcome.skipped!r}"
    )


def test_the_digest_never_drops_the_round_that_produced_the_reported_cause():
    """A plain head-slice keeps rounds 1-4 and discards the LAST round -- the one whose cause the
    attempt reports. The digest would then assert a cause and show no evidence for it, which is
    this REQ's own failure mode one layer down.

    Mutation M10: `_keep_head_and_tail` returns `items[:_ROUND_DIGEST_MAX_ROUNDS]`. RED here.
    """

    m = _harness()
    rounds = [{"round": i + 1, "skipped": None} for i in range(5)]
    rounds[-1]["skipped"] = "selection_or_planning_exception"

    kept = m._round_digest(rounds)

    assert len(kept) == m._ROUND_DIGEST_MAX_ROUNDS, "the digest must still be bounded"
    assert kept[-1]["round"] == 5, f"the last round must survive truncation; got {kept}"
    assert kept[-1]["skipped"] == "selection_or_planning_exception"

    # The same rule protects the attempt-level counterexample list, where a raising round's
    # evidence is appended LAST and exists nowhere else.
    atts = [
        {
            "skipped": "selection_or_planning_exception",
            "refinement_rounds": rounds,
            "counterexamples": [{"kind": f"early_{i}"} for i in range(4)]
            + [{"kind": "selection_or_planning_exception"}],
        }
    ]
    ces = m._induction_round_records(atts)[0]["counterexamples"]
    assert ces[-1]["kind"] == "selection_or_planning_exception", (
        f"the raising round's only evidence must survive truncation; got {ces}"
    )


def test_the_harness_channel_delta_clamps_and_survives_junk():
    """SCENARIO-ARC-WMTE-6710-7 requires the clamp, and the harness twin of the delta did not
    have it -- it returned -500 on a counter reset. It also runs AFTER the episode, so raising
    on a junk value would destroy a whole row of real data over a diagnostic field.

    Mutation M11: restore the unclamped `int(after[k]) - int(before[k])` comprehension. RED here.
    """

    m = _harness()

    assert m._channel_delta({"chars_final": 500}, {"chars_final": 0}) == {"chars_final": 0}
    assert m._channel_delta({}, {"chars_final": 10}) == {"chars_final": 10}
    # Junk values are dropped, not raised on.
    assert m._channel_delta({}, {"ok": 3, "junk": "abc", "nested": {}, "nan": float("nan")}) == {
        "ok": 3
    }
    assert m._channel_delta({}, {}) == {}

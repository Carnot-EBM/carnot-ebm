"""REQ-ARC-WMTE-6042 / SCENARIO-ARC-WMTE-6042-WRITE-COLLAPSE-INSTRUMENTATION.

WHAT THIS PROTECTS, AND WHY IT EXISTS.

The ARC CEGIS induction-refinement harness produced a null that was read as "feedback does not
help". That reading was WITHDRAWN: the instrument is broken, and one of the three defects is a
REFACTOR WRITE PATH THAT COLLAPSES WORKING ENGINES. Measured on the two shipped CEGIS shards,
using `prefix_accuracy` -- fit on the rows the model was SHOWN, with answers, a field both
prior artifacts recorded and neither read:

    induce   (round 1)    28/88  reach >0, ceiling 1.0
    refactor (rounds 2+)   4/160 reach >0, ceiling 0.125     (and 0.125 is exactly 1/8)
    Fisher OR 18.2, p = 1.0294417830164442e-10
    15 of 83 cells with an emitted refactor fall from a PERFECT 1.0 to 0.0

The mechanism is upstream of the emitted code: `refactor_prompt(game, vr)` takes only a game id
and a VerifyResult, so the prompt NEVER CONTAINS the engine it instructs the model to preserve,
and `_bounded_mismatches(limit=5)` shows at most five FAILING transitions and nothing that
passed. "REFACTOR ... while keeping the cases it already gets right" is unachievable by
construction, so every refactor round is a blind re-induction.

WHAT THE INSTRUMENTATION ADDS, AND WHY IT IS NOT A FIX. `accuracy` and `cell_recall` said the
resulting engine was bad. They could not say WHAT it became, and that difference decides the
repair: an engine predicting a plausible-but-wrong change is a modelling error, while an engine
that returns its input unchanged or raises on every row is a degenerate artefact of the write
path. These tests protect the fields that tell those apart. Nothing here changes a decision --
no acceptance, no retention, no selection reads a new field.

THE MEASUREMENT IS BEHAVIOURAL, NEVER SYNTACTIC, and both syntactic failure modes are real and
were observed on this corpus: an engine that never writes the literal `return grid` and is
nonetheless identity on every row it answers, and an engine that writes it a dozen times and is
NOT identity on the corpus it was induced from. `test_says_return_grid_but_is_not_identity` and
`test_identity_without_the_literal_return_grid` are those two cases.

EVIDENCE SAFETY: every test that runs the loop redirects the engine store to `tmp_path`. The
shard-reading tests OPEN `results/*.jsonl` READ-ONLY. Nothing here writes `results/**`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from carnot.agentic import arc_executable_world_model as e3
from carnot.agentic import arc_llm_reinduction as reinduction
from carnot.agentic.arc_executable_world_model import Transition, WorldModelVerifier
from carnot.agentic.arc_llm_reinduction import execute_bounded_llm_reinduction
from carnot.experiment_5760_cegis_refinement_induction_ab import _summarize_cell


REPO_ROOT = Path(__file__).resolve().parents[2]
SHARDS = (
    REPO_ROOT / "results/exp5760_cegis_refinement_induction_shard.jsonl",
    REPO_ROOT / "results/exp5766_gemma31b_cegis_refinement_shard.jsonl",
)

GAME = "wcol"


# =============================================================================================
# PART 1 -- THE EXACT REPRODUCTION.
#
# This is deliberately NOT a synthetic happy path. It re-derives, from the real shipped shards,
# every number the write-collapse diagnosis rests on. If a future change to the harness, the
# shards, or the reading of them moves any of these, this test says so.
# =============================================================================================


def _round_records() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    """Parse both shards as JSON and split rounds by action.

    Reads `prefix_accuracy` as a TOP-LEVEL key of the round record. It is parsed, never grepped:
    a `grep -o | head -1` over these files matches a field DESCRIPTION as readily as a value,
    and has already manufactured one fabricated defect in this investigation.
    """

    induce: list[dict[str, Any]] = []
    refactor: list[dict[str, Any]] = []
    nulls = {"induce": 0, "refactor": 0}
    for shard in SHARDS:
        for line in shard.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            for r in rec.get("rounds") or []:
                bucket = induce if r.get("action") == "induce" else refactor
                if r.get("prefix_accuracy") is None:
                    nulls["induce" if r.get("action") == "induce" else "refactor"] += 1
                    continue
                bucket.append({**r, "_game": rec["game"], "_trial": rec.get("trial")})
    return induce, refactor, nulls


def test_exact_reproduction_of_the_induce_vs_refactor_prefix_asymmetry():
    """The asymmetry, reproduced number for number from the real shards.

    MUTATION PROOF: this test is about the DATA, not the new code, so it is proven by altering
    the expected constants rather than by deleting a pattern -- change any single expected
    number below and it goes red immediately.
    """

    induce, refactor, _nulls = _round_records()

    ind_pos = sum(1 for r in induce if float(r["prefix_accuracy"]) > 0)
    ref_pos = sum(1 for r in refactor if float(r["prefix_accuracy"]) > 0)

    assert (ind_pos, len(induce)) == (28, 88), "induce arm must reproduce 28/88"
    assert (ref_pos, len(refactor)) == (4, 160), "refactor arm must reproduce 4/160"
    assert max(float(r["prefix_accuracy"]) for r in induce) == 1.0
    assert max(float(r["prefix_accuracy"]) for r in refactor) == 0.125

    # The refactor arm's ONLY non-zero value is 1/8: the four "successes" are one correct row
    # out of an eight-row prefix, not partial success.
    assert {float(r["prefix_accuracy"]) for r in refactor if float(r["prefix_accuracy"]) > 0} == {
        0.125
    }

    from scipy.stats import fisher_exact

    odds, p = fisher_exact([[ind_pos, len(induce) - ind_pos], [ref_pos, len(refactor) - ref_pos]])
    assert odds == pytest.approx(18.2, abs=0.05)
    assert p == pytest.approx(1.0294417830164442e-10, rel=1e-9)
    assert p < 1e-9


def test_exact_reproduction_missing_prefix_accuracy_is_not_counted_as_zero():
    """MISSING IS NOT ZERO -- and the reason each row is missing is recorded, not swallowed.

    40 of the 288 round records carry a null `prefix_accuracy`. Every one is a `proposer_failed`
    round, i.e. no engine was ever emitted. Counting them as zeros would inflate the refactor
    denominator from 160 to 171 and manufacture an asymmetry partly out of proposer failures.

    MUTATION PROOF: change the exclusion so nulls are counted as 0.0 and the 28/88 + 4/160
    assertion above goes red (it becomes 28/117 and 4/171).
    """

    induce, refactor, nulls = _round_records()
    assert nulls == {"induce": 29, "refactor": 11}
    assert sum(nulls.values()) == 40
    assert len(induce) + len(refactor) + sum(nulls.values()) == 288

    reasons = set()
    for shard in SHARDS:
        for line in shard.read_text().splitlines():
            if not line.strip():
                continue
            for r in json.loads(line).get("rounds") or []:
                if r.get("prefix_accuracy") is None:
                    reasons.add(r.get("skipped"))
                    assert r.get("proposer_ok") is False
    assert reasons == {"proposer_failed"}


def test_exact_reproduction_perfect_induce_rounds_collapse_to_zero():
    """15 of 83 cells fall from a PERFECT 1.0 to 0.0, and EVERY refactor round after a perfect
    induce is 0.0 -- 30 of 30, spanning 6 games. Not one game, not one generator.

    MUTATION PROOF: change 15 -> 14 or 30 -> 29 and this goes red.
    """

    collapsed_cells = 0
    cells_with_emitted_refactor = 0
    collapsed_games: set[str] = set()
    after_perfect: list[float] = []
    per_shard_positive: list[tuple[int, int]] = []

    for shard in SHARDS:
        s_ind = s_ref = 0
        for line in shard.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            rounds = rec.get("rounds") or []
            ind = [
                float(r["prefix_accuracy"])
                for r in rounds
                if r.get("action") == "induce" and r.get("prefix_accuracy") is not None
            ]
            ref = [
                float(r["prefix_accuracy"])
                for r in rounds
                if r.get("action") == "refactor" and r.get("prefix_accuracy") is not None
            ]
            s_ind += sum(1 for v in ind if v > 0)
            s_ref += sum(1 for v in ref if v > 0)
            if ref:
                cells_with_emitted_refactor += 1
                if ind and max(ind) == 1.0 and max(ref) == 0.0:
                    collapsed_cells += 1
                    collapsed_games.add(rec["game"])
            if ind and max(ind) == 1.0:
                after_perfect.extend(ref)
        per_shard_positive.append((s_ind, s_ref))

    assert (collapsed_cells, cells_with_emitted_refactor) == (15, 83)
    assert len(collapsed_games) == 6, "the collapse is not confined to a single game"
    assert len(after_perfect) == 30
    assert all(v == 0.0 for v in after_perfect), "not one refactor after a perfect induce survives"

    # INDEPENDENT IN BOTH SHARDS: two different generators, same direction. Rules out
    # "one bad generator" as the explanation.
    (ind_a, ref_a), (ind_b, ref_b) = per_shard_positive
    assert (ind_a, ref_a) == (9, 2)
    assert (ind_b, ref_b) == (19, 2)
    for ind_n, ref_n in per_shard_positive:
        assert ind_n > ref_n


def test_exact_reproduction_every_round_took_the_reject_path():
    """Not one round in either shard ever cleared the held-out gate.

    This is what makes the reject path the right instrumentation site: it covers 100% of the
    rounds that carry a `prefix_accuracy`, so the new fields are populated for every round the
    diagnosis is about, at zero additional engine calls.

    MUTATION PROOF: if a future harness change lets a round through the gate, the skip-reason
    set below gains a member and this goes red -- which is the signal to extend the
    instrumentation to the accept path rather than to silently under-cover it.
    """

    reasons: set[str] = set()
    snapshots_match = total = 0
    for shard in SHARDS:
        for line in shard.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            for r in rec.get("rounds") or []:
                reasons.add(r.get("skipped"))
            # `n_engine_snapshots == induce_loaded + n_refactor_emitted` in every cell is what
            # rules out an IMPORT failure: the engines were emitted and loaded, they were just
            # worthless.
            total += 1
            expected = int(rec["n_refactor_emitted"]) + (1 if rec["round1_loaded"] else 0)
            snapshots_match += int(rec["n_engine_snapshots"] == expected)

    assert reasons == {"heldout_transition_verification_failed", "proposer_failed"}
    assert (snapshots_match, total) == (117, 117), "collapse is not an import failure"


# =============================================================================================
# PART 2 -- THE INSTRUMENTATION CLASSIFIES THE REAL RESIDUE CLASSES.
#
# The residue population found on disk was: 3 of 10 strictly identity, one identity on every row
# it did not raise on, and 10 of 10 scoring ZERO exact transitions. The engines below are those
# shapes in miniature.
# =============================================================================================

_MOVES = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}


def _true_next(grid: np.ndarray, action: int) -> np.ndarray:
    g = grid.copy()
    pos = np.argwhere(g == 3)
    r, c = int(pos[0][0]), int(pos[0][1])
    dr, dc = _MOVES[int(action) % 4]
    g[r, c] = 0
    g[(r + dr) % g.shape[0], (c + dc) % g.shape[1]] = 3
    return g


_CORPUS_N = 12


def _corpus(n: int = _CORPUS_N) -> tuple[list[Transition], np.ndarray]:
    """A corpus where EVERY row genuinely changes state.

    n=12 exceeds `WorldModelVerifier.score`'s `max_mismatch` default of 8 on purpose, so the
    uncapped-raise-count test has something the capped `mismatches` list cannot express.
    """

    grid = np.zeros((6, 6), dtype=int)
    grid[2, 2] = 3
    root = grid.copy()
    rows: list[Transition] = []
    for i in range(n):
        action = i % 4
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


def _score(engine) -> Any:
    rows, _root = _corpus()
    return WorldModelVerifier(list(rows)).score(engine)


def test_strict_identity_engine_is_reported_as_functionally_identity():
    """The g50t / vc33 residue shape: returns its input, always.

    MUTATION PROOF: delete the `n_output_equals_input += 1` accumulator in
    `WorldModelVerifier.score` and `identity_rate` drops to 0.0 with `functionally_identity`
    False -- this test goes red.
    """

    vr = _score(lambda g, a, d: np.asarray(g).copy())

    assert vr.n_engine_called == 12
    assert vr.n_engine_raised == 0
    assert vr.n_output_equals_input == 12
    assert vr.identity_rate == 1.0
    assert vr.functionally_identity is True
    assert vr.identity_measurable is True
    # And it is worth exactly nothing: zero exact transitions, the 10-of-10 residue finding.
    assert vr.n_correct == 0
    assert vr.accuracy == 0.0


def test_says_return_grid_but_is_not_identity():
    """THE ANTI-SYNTACTIC PROOF, half one -- the misattributed-exemplar shape.

    An engine whose source says `return grid` on many branches and which nonetheless MUTATES on
    this corpus. A source-scanning identity check would call this identity; it is not. Only
    executing it settles the question.

    MUTATION PROOF: replace the behavioural comparison with a source scan for `return grid` and
    this test goes red.
    """

    src = """
import numpy as np


def engine(grid, action, data):
    g = np.asarray(grid).copy()
    if int(action) == 0:
        g[0, 0] = 7
        return g
    if int(action) == 1:
        g[0, 1] = 7
        return g
    if int(action) == 2:
        g[0, 2] = 7
        return g
    if int(action) == 3:
        g[0, 3] = 7
        return g
    return grid
"""
    assert src.count("return grid") == 1 and src.count("return g\n") >= 4
    ns: dict[str, Any] = {}
    exec(compile(src, "<not_identity>", "exec"), ns)  # noqa: S102 -- fixture engine

    vr = _score(ns["engine"])

    assert vr.n_engine_called == 12
    assert vr.n_output_equals_input == 0
    assert vr.functionally_identity is False, "writing `return grid` does not make it identity"
    assert vr.identity_measurable is True


def test_identity_without_the_literal_return_grid():
    """THE ANTI-SYNTACTIC PROOF, half two -- the sb26 shape.

    Identity on every row it answers, WITHOUT the source ever containing `return grid`. A
    source-scanning check would miss this entirely.

    MUTATION PROOF: as above -- any syntactic substitute for the behavioural comparison fails
    this test and its sibling simultaneously, in opposite directions.
    """

    src = """
import numpy as np


def engine(grid, action, data):
    out = np.asarray(grid).copy()
    return out
"""
    assert "return grid" not in src
    ns: dict[str, Any] = {}
    exec(compile(src, "<silent_identity>", "exec"), ns)  # noqa: S102 -- fixture engine

    vr = _score(ns["engine"])

    assert vr.functionally_identity is True
    assert vr.n_output_equals_input == 12


def test_identity_on_answered_rows_excludes_raised_rows_from_the_denominator():
    """The sb26 shape proper: identity on all 8 rows it answers, raising on the rest.

    The raised rows are in NEITHER numerator nor denominator -- a row that produced no output is
    evidence of neither identity nor non-identity -- and `engine_raise_rows` keeps the exclusion
    visible rather than silent.

    MUTATION PROOF: change the denominator to `n_engine_called` (i.e. stop excluding raises) and
    `identity_rate` becomes 8/12 = 0.667 with `functionally_identity` False -- red.
    """

    def engine(g, a, d):
        if int(a) == 3:
            raise ValueError("does not model this action")
        return np.asarray(g).copy()

    vr = _score(engine)

    assert vr.n_engine_called == 12
    assert vr.n_engine_raised == 3
    assert vr.n_output_equals_input == 9
    assert vr.identity_rate == 1.0, "identity on every row it ANSWERED"
    assert vr.functionally_identity is True
    assert vr.engine_raise_kinds == {"ValueError": 3}


def test_raise_count_is_uncapped_unlike_the_mismatch_sample():
    """`mismatches` stops at `max_mismatch`; the raise COUNT must not.

    A 12-row wipeout and an 8-row one are very different engines, and reading the count off the
    capped mismatch list cannot tell them apart. NO SILENT CENSORING.

    MUTATION PROOF: delete `n_engine_raised += 1` and derive the count from
    `len(vr.mismatches)` instead -- it caps at 8 and this goes red.
    """

    def engine(g, a, d):
        raise RuntimeError("total wipeout")

    vr = _score(engine)

    assert vr.n_engine_called == 12
    assert vr.n_engine_raised == 12
    assert len(vr.mismatches) == 8, "the mismatch SAMPLE is capped, as designed"
    assert vr.n_engine_raised > len(vr.mismatches)
    assert vr.engine_raise_kinds == {"RuntimeError": 12}


def test_an_engine_that_answers_nothing_is_not_reported_as_identity():
    """NON-VACUITY. UNMEASURABLE IS NOT CLEAN.

    With every row raising, `n_output_equals_input == answered` is trivially 0 == 0. Without the
    `> 0` guard that reads as "functionally identity", which is a claim the data cannot support.

    MUTATION PROOF: drop the `(n_engine_called - n_engine_raised) > 0` conjunct from
    `functionally_identity` and this test goes red.
    """

    def engine(g, a, d):
        raise RuntimeError("never answers")

    vr = _score(engine)

    assert vr.functionally_identity is False
    assert vr.identity_measurable is False
    assert vr.identity_rate == 0.0


def test_empty_corpus_is_unmeasurable_not_clean():
    """The same non-vacuity guard, reached from the other direction: nothing to score at all."""

    vr = WorldModelVerifier([]).score(lambda g, a, d: g)

    assert vr.n_engine_called == 0
    assert vr.functionally_identity is False
    assert vr.identity_measurable is False


def test_a_genuinely_wrong_but_non_degenerate_engine_is_not_flagged_identity():
    """THE CONTROL, AND IT IS NOT VACUOUS.

    An engine that models a real change and gets it WRONG must be distinguishable from a
    degenerate one -- that distinction is the entire point of the instrumentation. This control
    scores accuracy 0.0 exactly like the identity engine does, so `accuracy` alone cannot
    separate them and `functionally_identity` must.

    MUTATION PROOF: make `functionally_identity` return True unconditionally and this goes red
    while the identity tests above still pass -- which is what makes the pair non-vacuous.
    """

    def engine(g, a, d):
        out = np.asarray(g).copy()
        pos = np.argwhere(out == 3)
        r, c = int(pos[0][0]), int(pos[0][1])
        out[r, c] = 0
        out[(r + 1) % out.shape[0], (c + 1) % out.shape[1]] = 3  # always diagonal: wrong
        return out

    vr = _score(engine)

    assert vr.accuracy == 0.0, "as wrong as the identity engine, by the headline metric"
    assert vr.functionally_identity is False, "but NOT degenerate -- it predicts a real change"
    assert vr.n_output_equals_input == 0
    assert vr.identity_measurable is True


# =============================================================================================
# PART 3 -- TRAJECTORY INVARIANCE.
#
# This is instrumentation: it may only ADD recorded fields. These tests prove it changed nothing
# else -- structurally, not by inspection.
# =============================================================================================


def test_instrumentation_adds_zero_engine_calls():
    """THE LOAD-BEARING INVARIANCE PROOF, and it is structural rather than observational.

    An engine may hold module-level state, so an instrumentation pass that invoked the engine
    even once more could change what a LATER consumer in the same round observes. The
    accumulators ride the EXISTING per-transition loop, so the engine is called exactly once per
    scored row and no more. Proven by instrumenting the CALLEE and counting.

    MUTATION PROOF: implement the identity measurement as a second pass over the transitions
    (the obvious way to write it) and the count doubles -- this goes red.
    """

    calls: list[tuple[int, int]] = []

    def counting_engine(g, a, d):
        calls.append((int(a), int(np.asarray(g).sum())))
        return np.asarray(g).copy()

    rows, _root = _corpus()
    vr = WorldModelVerifier(list(rows)).score(counting_engine)

    assert len(calls) == len(rows), "exactly one engine call per row -- not one more"
    assert len(calls) == vr.n_engine_called


def test_instrumentation_does_not_mutate_the_recorded_transitions():
    """The engine is handed `t.grid.copy()`, so a mutating engine cannot corrupt the corpus.

    MUTATION PROOF: drop the `.copy()` in `score` and the before-grid comparison below goes red.
    """

    rows, _root = _corpus()
    before = [(r.grid.copy(), r.next_grid.copy()) for r in rows]

    def vandal_engine(g, a, d):
        g[:] = 9  # writes straight into whatever it was handed
        return g

    WorldModelVerifier(list(rows)).score(vandal_engine)

    for row, (grid_before, next_before) in zip(rows, before):
        assert np.array_equal(row.grid, grid_before)
        assert np.array_equal(row.next_grid, next_before)


# The round-row keys that existed BEFORE this instrumentation, on the reject path. Frozen here
# so that a change to any of them -- a rename, a removal -- is a test failure rather than a
# silent shard-schema break.
_PRE_EXISTING_REJECT_ROW_KEYS = {
    "round",
    "action",
    "proposer_ok",
    "message",  # set whenever the proposer returned one -- the scripted proposer always does
    "retention_signal_heldout_change_consistency",
    "retention_signal_true_changed_cells",
    "retained_as_best_engine",
    "selected_candidate_name",
    "goal_candidate_names",
    "dynamics_candidate_names",
    "prefix_accuracy",
    "heldout_accuracy",
    "heldout_threshold",
    "accepted_by_heldout_verifier",
    "trust_energy",
    "counterexample",
    "skipped",
}
_NEW_INSTRUMENTATION_ROW_KEYS = {
    "engine_behaviour_corpus",
    "engine_rows_scored",
    "engine_raise_rows",
    "engine_raise_kinds",
    "engine_output_equals_input_rows",
    "engine_identity_frac",
    "engine_functionally_identity",
    "engine_identity_measurable",
    "engine_source_sha256",
}
# Keys a SIBLING REQ is entitled to record on the same row. REQ-ARC-WMTE-6090 adds these under
# `CARNOT_ARC_CEGIS_ACCEPT_SPLIT=1` only. Listed explicitly rather than waved through with a
# wildcard, so a genuinely unexpected key still fails the containment check above.
_SIBLING_REQ_ROW_KEYS = {
    "acceptance_split_decidable",
    "acceptance_split_reason",
    "acceptance_split_gradeable_rows",
    "refinement_corpus_rows",
    # REQ-ARC-WMTE-6710: this round's per-channel character split. Record only; nothing branches
    # on it. Declared here because this guard is what makes a new round key a deliberate act.
    "channel_chars",
}

IDENTITY_SRC = """
import numpy as np


def engine(grid, action, data):
    return np.asarray(grid).copy()


def is_level_complete(grid):
    return False
"""

WRONG_BUT_LIVE_SRC = """
import numpy as np


def engine(grid, action, data):
    g = np.asarray(grid).copy()
    g[0, 0] = int(g[0, 0]) + 1
    return g


def is_level_complete(grid):
    return False
"""


class _ScriptedProposer:
    """Replays a fixed engine source per round, writing each into the store -- the real shape of
    the failure, in which `induce` writes round 1 and every `refactor` OVERWRITES it."""

    model_specs = "scripted-write-collapse-instrumentation-test-proposer"

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


def _run_loop(monkeypatch, tmp_path: Path, sources: list[str]):
    monkeypatch.setattr(e3, "E3_DIR", tmp_path)
    monkeypatch.setattr(reinduction, "MAX_REFINEMENT_ROUNDS", 3)
    rows, root = _corpus()
    proposer = _ScriptedProposer(tmp_path, GAME, sources)
    result = execute_bounded_llm_reinduction(
        game=GAME,
        transitions=rows,
        cell=1,
        root_grid=root,
        proposer=proposer,
        candidate_provider=lambda engine, goal: [("loaded_world_model.py", engine, goal)],
        load_engine=e3.load_engine,
        plan_in_model=lambda engine, goal, grid: None,
        max_rounds=3,
        min_heldout_accuracy=1.0,
    )
    return result, proposer


def test_round_rows_gain_only_the_new_keys(monkeypatch, tmp_path):
    """ADDITIVE ONLY: every pre-existing round key survives, and the delta is exactly the new set.

    MUTATION PROOF: rename or drop any instrumentation key and the exact set-equality below goes
    red; drop a PRE-EXISTING key and the subset assertion goes red.
    """

    result, proposer = _run_loop(monkeypatch, tmp_path, [IDENTITY_SRC, IDENTITY_SRC, IDENTITY_SRC])

    assert proposer.writes == [0, 1, 2], "the loop must really have run three rounds"
    reject_rows = [
        r for r in result.rounds if r.get("skipped") == "heldout_transition_verification_failed"
    ]
    assert reject_rows, "the scenario must reach the reject path -- otherwise it proves nothing"

    for row in reject_rows:
        keys = set(row)
        assert _PRE_EXISTING_REJECT_ROW_KEYS <= keys, (
            f"a pre-existing round key disappeared: {_PRE_EXISTING_REJECT_ROW_KEYS - keys}"
        )
        # CONTAINMENT, NOT EQUALITY -- and the difference is not pedantry, it was a live failure.
        # This assertion shipped as an exact set-equality against a CLOSED key set. REQ-ARC-WMTE-6090
        # then legitimately added four `acceptance_split_*` / `refinement_corpus_rows` keys to the
        # same row, under its own default-OFF flag, and this test went RED under
        # `CARNOT_ARC_CEGIS_ACCEPT_SPLIT=1` (1 failed, 21 passed) -- not because anything regressed,
        # but because a sibling REQ is allowed to extend a row this test had frozen shut. The
        # property actually being protected is "every pre-existing key survives AND every
        # instrumentation key arrives", which is containment in both directions; equality
        # additionally asserts "and nobody else may ever record anything here", which was never
        # this test's business to claim.
        assert _NEW_INSTRUMENTATION_ROW_KEYS <= keys, (
            f"an instrumentation key is missing: {_NEW_INSTRUMENTATION_ROW_KEYS - keys}"
        )
        foreign = keys - _PRE_EXISTING_REJECT_ROW_KEYS - _NEW_INSTRUMENTATION_ROW_KEYS
        assert foreign <= _SIBLING_REQ_ROW_KEYS, (
            f"an unexpected key appeared on the round row: {foreign - _SIBLING_REQ_ROW_KEYS}"
        )


def test_instrumentation_reads_the_scoring_pass_the_loop_already_ran(monkeypatch, tmp_path):
    """NO SECOND SCORING PASS -- asserted structurally, so it stays true.

    `engine_rows_scored` must equal the counterexample's `real_n`, because both come from the
    SAME already-computed `real_verify`. If anyone later re-implements the instrumentation as an
    independent scoring pass -- over a different corpus, or with extra engine calls -- these two
    numbers part company and this test says so.

    MUTATION PROOF: re-score inside the instrumentation block over `prefix` instead of reading
    `real_verify`, and the equality breaks.
    """

    # PIN THE FLAG OFF EXPLICITLY rather than inheriting whatever the ambient environment holds.
    # This test asserts the OFF-direction contract, so it must ESTABLISH OFF: run the suite with
    # `CARNOT_ARC_CEGIS_ACCEPT_SPLIT=1` exported and an inheriting version fails for a reason that
    # has nothing to do with the property under test, which is a flake, not a finding.
    monkeypatch.delenv("CARNOT_ARC_CEGIS_ACCEPT_SPLIT", raising=False)
    result, _proposer = _run_loop(monkeypatch, tmp_path, [IDENTITY_SRC])

    checked = 0
    for row in result.rounds:
        if "engine_rows_scored" not in row:
            continue
        checked += 1
        assert row["engine_rows_scored"] == row["counterexample"]["real_n"]
        # The flag is OFF here, so the corpus really is the full transition list.
        assert row["engine_behaviour_corpus"] == "full_transitions"
        assert row["engine_rows_scored"] == _CORPUS_N
    assert checked >= 1


def test_the_corpus_label_names_the_rows_actually_scored_under_the_sibling_split(
    monkeypatch, tmp_path
):
    """THE LABEL MUST BE DERIVED FROM THE ROWS, NOT ASSERTED AS A LITERAL.

    `engine_behaviour_corpus` shipped as a hardcoded `"full_transitions"`. That was true when it
    was written and became FALSE the moment REQ-ARC-WMTE-6090 made `refinement_corpus` conditional
    in the same function: with `CARNOT_ARC_CEGIS_ACCEPT_SPLIT=1` the verifier is handed the full
    corpus MINUS the reserved acceptance block, and the field went on claiming otherwise. Nothing
    raised; the record simply described a denominator that was not its own.

    The failure this guards is not "wrong string". It is that the ONE field whose entire job is to
    say WHICH ROWS a number was computed over can silently stop matching them, in an artifact whose
    stated purpose is denominator honesty.

    MUTATION PROOF: restore the literal `row["engine_behaviour_corpus"] = "full_transitions"` and
    this goes red on the label; neuter the split so the corpus is not actually narrowed and it goes
    red on the strict row-count inequality instead. The sibling test above pins the OFF direction,
    so the pair fails in opposite directions and neither can be satisfied by a constant.
    """

    monkeypatch.setenv("CARNOT_ARC_CEGIS_ACCEPT_SPLIT", "1")
    result, _proposer = _run_loop(monkeypatch, tmp_path, [IDENTITY_SRC, IDENTITY_SRC])

    checked = 0
    for row in result.rounds:
        if "engine_rows_scored" not in row:
            continue
        checked += 1
        # The corpus really is narrower -- otherwise this test proves nothing about the label.
        assert row["engine_rows_scored"] < _CORPUS_N, (
            "the acceptance split did not withhold any row, so this is a vacuous control"
        )
        assert row["engine_rows_scored"] == row["refinement_corpus_rows"]
        assert row["engine_behaviour_corpus"] == "refinable_minus_acceptance"
        # ...and it still reads off the SAME already-computed scoring pass, no second engine run.
        assert row["engine_rows_scored"] == row["counterexample"]["real_n"]
    assert checked >= 1


def test_the_counterexample_handed_to_refactor_is_untouched(monkeypatch, tmp_path):
    """THE PROMPT MUST NOT CHANGE.

    `last_counterexample` flows into `_counterexample_result` and from there into
    `refactor_prompt`. A field added THERE would change the prompt, hence the completion, hence
    the trajectory. The instrumentation therefore writes to `row` only, and the counterexample
    keeps exactly its historical key set.

    MUTATION PROOF: add any instrumentation key to `last_counterexample` and this goes red.
    """

    result, _proposer = _run_loop(monkeypatch, tmp_path, [IDENTITY_SRC, WRONG_BUT_LIVE_SRC])

    seen = 0
    for row in result.rounds:
        ce = row.get("counterexample")
        if not ce or ce.get("kind") != "heldout_transition_verification_failed":
            continue
        seen += 1
        assert set(ce) == {
            "kind",
            "selected_candidate_name",
            "heldout_accuracy",
            "heldout_threshold",
            "real_n",
            "real_n_correct",
            "real_accuracy",
            "real_mismatches",
        }
        assert not any(k.startswith("engine_") for k in ce)
    assert seen >= 1


def test_the_loop_records_the_degenerate_engine_it_actually_wrote(monkeypatch, tmp_path):
    """END TO END: a refactor that writes an identity engine is RECORDED as having done so.

    This is the write-collapse signature the shards could not show, because the loop never
    recorded it and the shard whitelist would have dropped it anyway.

    MUTATION PROOF: delete the `row["engine_functionally_identity"] = ...` line and this goes red.
    """

    result, _proposer = _run_loop(
        monkeypatch, tmp_path, [WRONG_BUT_LIVE_SRC, IDENTITY_SRC, IDENTITY_SRC]
    )

    rows = [r for r in result.rounds if "engine_functionally_identity" in r]
    assert len(rows) >= 2

    first, later = rows[0], rows[-1]
    assert first["action"] == "induce"
    assert first["engine_functionally_identity"] is False, "round 1 wrote a live engine"
    assert later["action"] == "refactor"
    assert later["engine_functionally_identity"] is True, "the refactor collapsed it to identity"
    assert later["engine_identity_frac"] == 1.0
    assert later["engine_raise_rows"] == 0


def test_round_level_engine_provenance_distinguishes_the_rounds(monkeypatch, tmp_path):
    """A residue can now be tied to a ROUND, not merely to a cell.

    The engine store is one mutable path per game, so before this fingerprint the on-disk file
    could not be attributed to a round at all -- `_read_engine_source` was called only on the
    retention path, i.e. never on the rounds that collapse.

    MUTATION PROOF: delete the `engine_source_sha256` entry from the `row.update` block and this
    goes red.
    """

    result, _proposer = _run_loop(
        monkeypatch, tmp_path, [WRONG_BUT_LIVE_SRC, IDENTITY_SRC, IDENTITY_SRC]
    )

    prints = [r["engine_source_sha256"] for r in result.rounds if "engine_source_sha256" in r]
    assert len(prints) >= 3
    assert all(p is not None for p in prints)
    assert prints[0] != prints[1], "round 2 overwrote round 1's file with different bytes"
    assert prints[1] == prints[2], "rounds 2 and 3 wrote identical bytes"
    assert all(set(p) == {"sha256_16", "chars"} for p in prints)
    assert prints[0]["chars"] == len(WRONG_BUT_LIVE_SRC)


def test_missing_engine_source_is_none_not_an_empty_hash(monkeypatch, tmp_path):
    """MISSING IS NOT ZERO, at the fingerprint.

    An absent store must read as MISSING, never as "an engine with no bytes".

    MUTATION PROOF: return a hash of the empty string instead of None and this goes red.
    """

    from carnot.agentic.arc_llm_reinduction import _engine_source_fingerprint

    monkeypatch.setattr(e3, "E3_DIR", tmp_path)
    assert _engine_source_fingerprint("no_such_game_at_all") is None


# =============================================================================================
# PART 4 -- THE SHARD WHITELIST.
#
# `_summarize_cell`'s `slim_rounds` is a WHITELIST. An upstream field that is not named there is
# dropped before it reaches the shard, silently. That is the half of this fix that would
# otherwise make the other half a no-op.
# =============================================================================================


def _cell(rounds: list[dict[str, Any]]) -> dict[str, Any]:
    return _summarize_cell(
        game=GAME,
        trial=0,
        outcome=type(
            "O",
            (),
            {
                "rounds": rounds,
                "planned": False,
                "refinement_rounds_used": len(rounds),
                "heldout_accuracy": 0.0,
                "skipped": "",
            },
        )(),
        sources=["src"],
        coord_set=set(),
        stop_log=[],
        raw_len_log=[],
        wall_s=1.0,
        err=None,
    )


def test_the_shard_whitelist_carries_the_new_fields():
    """Without this, the instrumentation reaches the loop and dies before the shard.

    MUTATION PROOF: remove any one of the new keys from `slim_rounds` and this goes red -- which
    is exactly the failure that would otherwise be invisible, because the loop would still be
    recording the field correctly.
    """

    row = {
        "round": 2,
        "action": "refactor",
        "proposer_ok": True,
        "heldout_accuracy": 0.0,
        "prefix_accuracy": 0.0,
        "skipped": "heldout_transition_verification_failed",
        "engine_behaviour_corpus": "full_transitions",
        "engine_rows_scored": 12,
        "engine_raise_rows": 3,
        "engine_raise_kinds": {"ValueError": 3},
        "engine_output_equals_input_rows": 9,
        "engine_identity_frac": 1.0,
        "engine_functionally_identity": True,
        "engine_identity_measurable": True,
        "engine_source_sha256": {"sha256_16": "abc123", "chars": 42},
    }

    slim = _cell([row])["rounds"][0]

    for key in _NEW_INSTRUMENTATION_ROW_KEYS:
        assert key in slim, f"{key} was dropped by the slim_rounds whitelist"
        assert slim[key] == row[key], f"{key} was mangled on the way to the shard"


def test_an_unmeasured_round_reaches_the_shard_as_none_not_zero():
    """MISSING IS NOT ZERO, at the shard boundary.

    An ACCEPTED round takes a path with no free scoring pass, so it carries no engine-behaviour
    record. It must arrive as None -- read as NOT MEASURED -- and never as "0 rows, not
    identity", which would be a fabricated measurement.

    MUTATION PROOF: default the whitelist entries to 0 / False instead of `.get(...)` and this
    goes red.
    """

    slim = _cell(
        [
            {
                "round": 1,
                "action": "induce",
                "proposer_ok": True,
                "heldout_accuracy": 1.0,
                "prefix_accuracy": 1.0,
            }
        ]
    )["rounds"][0]

    for key in _NEW_INSTRUMENTATION_ROW_KEYS:
        assert slim[key] is None, f"{key} must be None (not measured), never a fabricated zero"

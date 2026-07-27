"""REQ-ARC-WMTE-6012 -- the HIDDEN-STATE trust gate's blind spot, locked as a regression.

CONTEXT. REQ-ARC-WMTE-6011 shipped `change_gate_decision`, wired into the NON-hidden-state
branch of `arc_competition_agent._induce_and_plan`. The other branch -- taken for the 11
`HIDDEN_STATE_GAME_IDS`, which include every one of the 0.08-wall games
(cn04/ar25/sc25/sk48/wa30) -- never calls it. It admits on `trust_pass`, defined by
`arc_world_model_trust_energy.score_change_weighted_consistency` as

    trust_pass := nondegenerate AND consistency >= 0.5
    consistency := correct_changed_cells / true_changed_cells

`consistency` masks to TRULY-CHANGED cells, so it is recall and cannot see a cell the engine
wrote that reality never changed. That module's docstring states this limit; these tests
MEASURE it, which is a different thing. exp6011's artifact, its script, and its 25-test
suite call `select_trusted_world_model` exactly zero times.

HERMETIC on purpose, following the sibling suite: fixtures in memory, no game sim, no
network. The real-corpus evidence (33 matched rows over 11 hidden-state games x 3 seeds)
lives in `scripts/experiments/experiment_6012_hidden_state_trust_gate_hole.py` and its
artifact. What is asserted here is the STRUCTURE of the defect, so a regression fails
immediately rather than at the next survey.

These tests assert the RELATIONSHIPS (gate A admits what gate B rejects; metric X is
identical where metric Y differs), never hard-coded corpus constants, so a re-tune of a
threshold cannot quietly empty the pass region and still leave the suite green.
"""

from __future__ import annotations

import numpy as np

from carnot.agentic.arc_executable_world_model import (
    Transition,
    WorldModelVerifier,
    change_gate_decision,
)
from carnot.agentic.arc_world_model_trust_energy import (
    HIDDEN_STATE_GAME_IDS,
    WorldModelCandidate,
    score_change_weighted_consistency,
    select_trusted_world_model,
)

H, W = 6, 6


def _corpus(n_change: int = 12, n_noop: int = 8) -> list[Transition]:
    """Changing transitions plus true no-ops -- the shape every real ARC corpus has.

    Each changing transition has a DISTINCT predecessor, so the corpus is deterministic and
    a correct engine exists. Without that, no engine could be right and every "does the gate
    admit a good engine" probe would be unfalsifiable for the wrong reason.

    INTERLEAVED, not concatenated. `select_trusted_world_model` holds out the LAST THIRD, so
    a corpus with all the changing transitions first leaves a held-out tail of pure no-ops:
    `true_changed_cells` is 0, consistency is 0/max(1,0) = 0.0, and the gate rejects even a
    perfect engine. The first version of this fixture did exactly that and produced three
    confidently-wrong failures. The ordering is part of the fixture's contract, not an
    incidental detail.
    """

    out: list[Transition] = []
    noop_every = max(1, round((n_change + n_noop) / max(1, n_noop))) if n_noop else 0
    n_made = 0
    for i in range(n_change):
        g0 = np.zeros((H, W), dtype=np.int16)
        g0[0, 0] = i + 1  # distinct predecessor
        g1 = g0.copy()
        g1[1 + (i % 4), 1 + (i // 4) % 4] = 7  # a real mechanic
        out.append(Transition(g0, 1, None, g1, 0, 0))
        if n_noop and n_made < n_noop and (i + 1) % noop_every == 0:
            g = np.zeros((H, W), dtype=np.int16)
            g[5, 5] = n_made + 1  # distinct, and unchanged
            out.append(Transition(g, 2, None, g.copy(), 0, 0))
            n_made += 1
    for j in range(n_made, n_noop):
        g = np.zeros((H, W), dtype=np.int16)
        g[5, 5] = j + 1
        out.append(Transition(g, 2, None, g.copy(), 0, 0))
    return out


def _honest(corpus):
    """Correct on this corpus, and correct in the ONLY way that matters here: it leaves true
    no-ops alone. Replay-based, which is fine for the arms below because every arm is the
    SAME replay plus one named perturbation -- the comparison isolates the perturbation."""

    table = {(t.grid.tobytes(), t.action): t.next_grid for t in corpus}

    def _engine(grid, action, data):
        return np.asarray(table.get((np.asarray(grid).tobytes(), action), np.asarray(grid)))

    return _engine


def _plus_spurious(corpus):
    """Honest, plus ONE cell reality never wrote. Invisible to any recall-style metric."""

    base = _honest(corpus)

    def _engine(grid, action, data):
        g = np.asarray(base(grid, action, data)).copy()
        g[H - 1, 0] = 999
        return g

    return _engine


def _plus_noop_hallucination(corpus):
    """Honest on real changes, invents a change on every TRUE NO-OP."""

    base = _honest(corpus)

    def _engine(grid, action, data):
        g = np.asarray(base(grid, action, data)).copy()
        if np.array_equal(g, np.asarray(grid)):
            g[H // 2, W // 2] = 998
        return g

    return _engine


def _identity(grid, action, data):
    return np.asarray(grid)


def _live_trust_pass(corpus, engine) -> bool:
    """The LIVE hidden-state admission boolean, via the production selector.

    One-candidate pool -> `selected_score` IS this engine. Calling the shipped function
    rather than re-deriving its formula is deliberate: two independent reimplementations of
    one wrong formula agreeing with each other is not evidence about the system.
    """

    sel = select_trusted_world_model(
        list(corpus), [WorldModelCandidate(name="probe", engine=engine)], hidden_state=True
    )
    return bool(sel.selected_score.trust_pass)


def _req6011(corpus, engine) -> dict:
    return change_gate_decision(WorldModelVerifier(list(corpus)).score(engine), enabled=True)


# ---------------------------------------------------------------------------
# The branch actually exists and covers the games that matter.
# ---------------------------------------------------------------------------


def test_the_0_08_wall_games_all_route_through_the_hidden_state_branch():
    """If this fails, the hole below stopped mattering -- or started mattering more."""

    for game in ("cn04", "ar25", "sc25", "sk48", "wa30"):
        assert game in HIDDEN_STATE_GAME_IDS
    # dc22, the game exp6011's must-not-fire control was hand-written for, is ALSO on this
    # branch -- so that control was validated against a gate its own game never reaches.
    assert "dc22" in HIDDEN_STATE_GAME_IDS


# ---------------------------------------------------------------------------
# THE HOLE.
# ---------------------------------------------------------------------------


def test_live_hidden_state_gate_admits_a_spurious_writer_that_req6011_rejects():
    """THE DEFECT, end to end: same corpus, same engine, two gates, opposite verdicts."""

    corpus = _corpus()
    engine = _plus_spurious(corpus)
    assert _live_trust_pass(corpus, engine) is True, "live gate should (wrongly) admit it"
    decision = _req6011(corpus, engine)
    assert decision["passed"] is False
    assert decision["spurious_changed_cells"] > 0


def test_live_hidden_state_gate_admits_a_noop_hallucinator_that_req6011_rejects():
    """The second attack: right about every real change, invents one on every no-op.

    `plan_in_model` walks the engine forward, so this engine hallucinates a transition at
    every step of every plan -- yet the live gate's consistency never looks at a no-op.
    """

    corpus = _corpus()
    engine = _plus_noop_hallucination(corpus)
    assert _live_trust_pass(corpus, engine) is True
    decision = _req6011(corpus, engine)
    assert decision["passed"] is False
    assert decision["reason"] == "engine_hallucinates_changes_on_noop_transitions"


def test_the_live_gates_consistency_is_literally_blind_to_both_attacks():
    """The MECHANISM of the hole, not just its effect.

    Both attacks score EXACTLY the honest engine's consistency -- not approximately, not on
    average. That equality is what "recall cannot see a spurious write" means, and asserting
    it directly is what makes the diagnosis a measurement instead of a story.
    """

    corpus = _corpus()
    honest = score_change_weighted_consistency(corpus, _honest(corpus))
    for attack in (_plus_spurious(corpus), _plus_noop_hallucination(corpus)):
        got = score_change_weighted_consistency(corpus, attack)
        assert got.consistency == honest.consistency
        assert got.nondegenerate == honest.nondegenerate
        assert got.trust_pass == honest.trust_pass


def test_req6011_union_fidelity_sees_the_spurious_write_that_consistency_cannot():
    """The asymmetry claim, demonstrated on one corpus: identical under recall, strictly
    worse under the symmetric union."""

    corpus = _corpus()
    honest_fid = _req6011(corpus, _honest(corpus))["change_fidelity"]
    spurious_fid = _req6011(corpus, _plus_spurious(corpus))["change_fidelity"]
    assert (
        score_change_weighted_consistency(corpus, _plus_spurious(corpus)).consistency
        == score_change_weighted_consistency(corpus, _honest(corpus)).consistency
    )
    assert spurious_fid < honest_fid


# ---------------------------------------------------------------------------
# MUST-NOT-FIRE. A gate that rejects everything is not an improvement over a gate
# that admits identity engines.
# ---------------------------------------------------------------------------


def test_both_gates_admit_the_honest_engine_on_the_same_corpus():
    corpus = _corpus()
    honest = _honest(corpus)
    assert _live_trust_pass(corpus, honest) is True
    assert _req6011(corpus, honest)["passed"] is True


def test_identity_engine_is_rejected_by_both_gates():
    """The origin incident, checked on THIS branch too. The hidden-state gate does NOT have
    the identity hole -- an honest negative worth pinning, because it bounds the defect: the
    branch is blind to spurious writes, not to inaction."""

    corpus = _corpus()
    assert _live_trust_pass(corpus, _identity) is False
    assert _req6011(corpus, _identity)["passed"] is False


# ---------------------------------------------------------------------------
# WHERE REQ-6011 ITSELF RUNS OUT -- measured on real re86 transitions, reproduced here.
# ---------------------------------------------------------------------------


def test_noop_channel_is_structurally_dead_when_every_transition_changes():
    """re86's shape: 40/40 held-out transitions change the grid, so `n_noop == 0`.

    `noop_hallucination_rate` then reports 0.0 via its `if n_noop else 0.0` fallback -- the
    value that means "perfectly clean" is also the value that means "not measurable". A gate
    condition cannot fire on this corpus, and the witness does not say so. That is the
    dead-channel shape this project has been bitten by before (a census once found 877 stat
    blocks with an `errors` key and zero non-zero values), so it is pinned here rather than
    left to be rediscovered.
    """

    corpus = _corpus(n_change=12, n_noop=0)
    decision = _req6011(corpus, _plus_noop_hallucination(corpus))
    assert decision["n_noop"] == 0
    assert decision["noop_hallucination_rate"] == 0.0
    assert decision["noop_ok"] is True  # cannot fire
    # ...and with no no-ops to perturb, the attack collapses into the honest engine, so it
    # is admitted -- correctly. The escape below is the one that is NOT correct.
    assert decision["passed"] is True


def test_a_spurious_writer_escapes_req6011_when_changes_are_dense():
    """THE MEASURED ESCAPE (re86, all 3 seeds): union fidelity is an ABSOLUTE threshold, and
    one spurious write per transition is a small RELATIVE decrement.

    With 1296-1742 genuinely-changed cells the spurious write costs ~0.08-0.12 of fidelity,
    so an honest-plus-spurious engine lands near 0.9 and clears 0.5 comfortably. The gate
    catches this attack on dc22 ONLY because dc22's honest engine already sits at ~0.5 with
    no headroom to give away -- i.e. union fidelity's sensitivity to spurious writes is
    proportional to how BAD the engine already is, which is backwards.
    """

    # Change-dense corpus: many changed cells per transition, no no-ops.
    corpus: list[Transition] = []
    for i in range(12):
        g0 = np.zeros((H, W), dtype=np.int16)
        g0[0, 0] = i + 1
        g1 = g0.copy()
        g1[2:5, 1:5] = 7  # a large, genuine change -> big union denominator
        corpus.append(Transition(g0, 1, None, g1, 0, 0))

    decision = _req6011(corpus, _plus_spurious(corpus))
    assert decision["spurious_changed_cells"] > 0, "the engine really does write spuriously"
    assert decision["passed"] is True, "and REQ-6011 admits it anyway -- the escape"


def test_the_proposed_fourth_channel_separates_where_union_fidelity_does_not():
    """THE PROPOSAL, calibrated from data rather than taste.

    `spurious_changed_cells` is ALREADY computed and ALREADY in the witness dict; it is just
    not a gate condition. Normalised PER CHANGING TRANSITION it is scale-free in the change
    density that defeats union fidelity. Measured over 33 real matched rows (11 hidden-state
    games x 3 seeds): honest engines max 0.0714, spurious attack min 1.0000 -- no overlap, a
    ~14x gap. Any threshold strictly between them separates every row; 0.25 is proposed for
    being well clear of BOTH, the same reasoning already used for
    WORLD_MODEL_MAX_NOOP_HALLUCINATION_RATE.

    Asserted as a SEPARATION, not against the constant 0.25, so re-tuning the constant
    cannot leave this green while emptying the pass region.
    """

    corpus: list[Transition] = []
    for i in range(12):
        g0 = np.zeros((H, W), dtype=np.int16)
        g0[0, 0] = i + 1
        g1 = g0.copy()
        g1[2:5, 1:5] = 7
        corpus.append(Transition(g0, 1, None, g1, 0, 0))

    def rate(engine) -> float:
        d = _req6011(corpus, engine)
        return float(d["spurious_changed_cells"]) / max(1, int(d["n_changing"]))

    honest_rate = rate(_honest(corpus))
    attack_rate = rate(_plus_spurious(corpus))
    assert honest_rate < attack_rate
    # And it catches precisely the case union fidelity misses on this same corpus.
    assert _req6011(corpus, _plus_spurious(corpus))["passed"] is True
    assert attack_rate > 0.25 >= honest_rate


def test_spurious_changed_cells_conflates_invented_writes_with_ordinary_error():
    """THE LIMITATION OF THE PROPOSAL ABOVE, pinned so it is not overclaimed.

    `spurious_changed_cells` is `(wrote & ~correct)` -- cells the engine CHANGED and got
    WRONG -- accumulated over changing transitions only. That is two different things at
    once: (a) the engine wrote where reality did not change (a genuinely invented
    transition), and (b) the engine wrote where reality DID change but predicted the wrong
    value (ordinary prediction error). Only (a) is a spurious write.

    This matters for calibration honesty. The 0.1875-vs-1.0 separation measured in exp6012
    has near-perfect lookup engines on its honest side, so it barely exercises (b). A
    genuinely IMPERFECT induced model -- the thing the gate actually has to judge -- accrues
    (b) in normal operation and could exceed a 0.25 threshold while inventing nothing. The
    engine below is wrong ONLY on cells reality genuinely changed and invents nothing at all,
    yet still registers spurious cells, which is the proof.

    The purer quantity is `(wrote & ~truly_changed)`, which the verifier does not currently
    compute. Recommending the threshold without this caveat would be calibrating against a
    population that does not represent the deployment population.
    """

    corpus = _corpus(n_change=12, n_noop=6)

    def wrong_value_but_never_invents(grid, action, data):
        """Changes exactly the cells reality changes -- to the wrong value."""

        g = np.asarray(grid).copy()
        # Find what reality did for this exact predecessor, then write a wrong value THERE.
        for t in corpus:
            if np.array_equal(np.asarray(t.grid), g) and t.action == action:
                changed = np.asarray(t.grid) != np.asarray(t.next_grid)
                g[changed] = 123  # wrong, but only where reality genuinely changed
                return g
        return g

    d = _req6011(corpus, wrong_value_but_never_invents)
    assert d["n_changing"] > 0
    # It invents nothing: every no-op is left exactly alone.
    assert d["noop_hallucination_rate"] == 0.0
    # ...and yet it is charged spurious cells, because the field counts wrong writes too.
    assert d["spurious_changed_cells"] > 0

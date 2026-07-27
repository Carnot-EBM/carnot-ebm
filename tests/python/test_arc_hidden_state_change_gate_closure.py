"""REQ-ARC-WMTE-6013 -- the change gate's HIDDEN-STATE branch coverage hole, closed.

WHAT THIS FILE GUARDS, AND WHY IT IS SEPARATE FROM test_arc_world_model_change_gate.py
======================================================================================
REQ-6011 shipped `change_gate_decision` wired into exactly ONE of E3AgentPolicy's two
admission branches. The other branch -- taken for the 11 HIDDEN_STATE_GAME_IDS, which
include every one of the 0.08-wall games -- admitted on `trust_pass` and never called the
change gate at all. exp6012 measured the consequence: an engine correct on every real
change that ALSO writes cells reality never wrote is admitted on 31/33 rows, the SAME rows
where the honest engine is admitted, because the incumbent `consistency` masks to true
changes only and is arithmetically blind to an invented write.

These tests are HERMETIC except where the requirement is specifically about REAL data.
The one non-hermetic test is `test_the_real_ondisk_lp85_engine_is_rejected_...`, and it is
non-hermetic ON PURPOSE: this project has twice shipped a guard that did not fire on its
own origin incident, and both times the guard had been validated against a fixture that
reproduced what the author BELIEVED the incident was. A fixture cannot discharge "would
this have caught the real thing"; only the real file can.

MUTATION PROOFS. Every test here was verified to FAIL when its own repair is reverted --
see the `MUTATION:` line on each. Those lines are not decoration: a test that passes both
with and without the change under test is not testing the change.
"""

from __future__ import annotations

import numpy as np
import pytest

from carnot.agentic.arc_executable_world_model import (
    SUBMITTED_WORLD_MODEL_CHANGE_GATE_HIDDEN_STATE_ENABLED,
    Transition,
    WorldModelVerifier,
    change_gate_decision,
    world_model_change_gate_hidden_state_enabled,
)
from carnot.agentic.arc_world_model_trust_energy import (
    WorldModelCandidate,
    select_trusted_world_model,
)

# MODULE-LEVEL on purpose, same reasoning as test_arc_world_model_change_gate.py's header.
# The origin-incident test loads a real game sim, which costs ~660MB. Paid inside a test
# body, that lands in conftest's PytestMemoryWatchdog setup->teardown delta and surfaces as
# a TEARDOWN ERROR -- an invisible failure of exactly the kind this project's testing
# discipline forbids. Paying it at COLLECTION time makes the per-test delta ~0. The
# alternative (a `memory_watchdog_skip` marker) would silence the watchdog rather than fix
# the cause, and a skipped or silenced test is an invisible failure.
_ORIGIN_CORPUS: dict = {}
try:  # pragma: no cover - exercised by the origin-incident test's assertions below
    from pathlib import Path as _Path

    from carnot.agentic import arc_executable_world_model as _e3

    _REPO = _Path(__file__).resolve().parents[2]
    if (_REPO / "results/arc_e3/lp85/world_model.py").exists():
        _engine, _ = _e3.load_engine("lp85")
        _trans, _ = _e3.collect_transitions("lp85", n=120, seed=0)
        _ORIGIN_CORPUS = {"engine": _engine, "transitions": list(_trans)}
    else:
        _ORIGIN_CORPUS = {"error": "results/arc_e3/lp85/world_model.py is missing from disk"}
except Exception as _exc:  # noqa: BLE001 - recorded and asserted on, never swallowed
    _ORIGIN_CORPUS = {"error": f"{type(_exc).__name__}: {_exc!r}"[:300]}

H, W = 8, 8


def _grid(avatar: tuple[int, int], counter: int) -> np.ndarray:
    """A grid with a real mechanic (the avatar) and a HUD-ish counter in the last row."""

    g = np.zeros((H, W), dtype=np.int16)
    g[avatar] = 3
    g[H - 1, 0] = counter % 10
    return g


def _noop_heavy_corpus(n_change: int = 12, n_noop: int = 36) -> list[Transition]:
    """The GAP-WM-TRUST-GATE shape: a minority of real changes among many no-ops.

    This is what lets an identity engine clear `accuracy >= 0.5` -- the defect the whole
    requirement exists to remove. Changes come FIRST and no-ops after, because
    `_split_prefix_heldout` takes the LAST third as held-out; interleaving matters and a
    corpus whose held-out tail is all no-ops would make every gate read
    `no_changing_transitions` and prove nothing (that is exactly what ft09's real corpus
    does, see the exp6013 artifact's VACUOUS set).
    """

    rows: list[Transition] = []
    total = n_change + n_noop
    # Spread the changes EVENLY across the whole corpus rather than front-loading them. The
    # first version of this fixture emitted all 12 changes in the first 24 rows, which left
    # the held-out last third entirely no-op -- every gate then read `no_changing_
    # transitions` and three tests failed on their own premise. That is the same vacuity
    # that makes the REAL ft09 corpus useless as evidence (0 of 40 held-out transitions
    # change), so getting it wrong here would have hidden the very defect being tested.
    every = max(1, total // max(1, n_change))
    for i in range(total):
        if i % every == 0:
            a = (1 + (i % 4), 1)
            b = (1 + (i % 4), 2)
            rows.append(Transition(_grid(a, i), 1, None, _grid(b, i + 1), 0, 0))
        else:
            a = (5, 5)
            rows.append(Transition(_grid(a, i), 2, None, _grid(a, i), 0, 0))
    return rows


def _perfect(grid, action, data):
    """Reproduces the corpus mechanic exactly: action 1 steps the avatar right, else idle."""

    g = np.asarray(grid).copy()
    if action == 1:
        where = np.argwhere(g == 3)
        if len(where):
            r, c = where[0]
            g[r, c] = 0
            g[r, min(c + 1, W - 1)] = 3
            g[H - 1, 0] = (int(g[H - 1, 0]) + 1) % 10
    return g


def _identity(grid, action, data):
    return np.asarray(grid).copy()


def _perfect_plus_invention(grid, action, data):
    """Correct on every real change, PLUS one write reality never makes.

    This is the exp6012 attack in miniature. The incumbent `consistency` scores it EXACTLY
    equal to `_perfect` because it masks to truly-changed cells and cell (0, W-1) is never
    among them. The symmetric union metric charges for it.
    """

    g = np.asarray(_perfect(grid, action, data)).copy()
    g[0, W - 1] = 9
    return g


def _candidate(engine, name="c"):
    return WorldModelCandidate(name=name, engine=engine, is_level_complete=lambda g: False)


def _select(engine):
    return select_trusted_world_model(
        _noop_heavy_corpus(), [_candidate(engine)], hidden_state=True
    ).selected_score


# ---------------------------------------------------------------------------
# The flag: default-off, and resolvable per arm.
# ---------------------------------------------------------------------------


def test_hidden_state_gate_ships_default_off_following_the_req6011_flag(monkeypatch):
    """MUTATION: change the default in `world_model_change_gate_hidden_state_enabled` to
    True and this fails -- which is the whole operator-facing claim of the change."""

    monkeypatch.delenv("CARNOT_ARC_WM_CHANGE_GATE_HIDDEN_STATE", raising=False)
    monkeypatch.delenv("CARNOT_ARC_WM_CHANGE_GATE", raising=False)
    assert SUBMITTED_WORLD_MODEL_CHANGE_GATE_HIDDEN_STATE_ENABLED is None
    assert world_model_change_gate_hidden_state_enabled() is False


def test_hidden_state_flag_follows_req6011_so_the_gate_arm_is_not_a_noop_on_wall_games(
    monkeypatch,
):
    """The 11 hidden-state games are every 0.08-wall game. If turning the change gate on
    left them untouched, the gate arm would measure nothing on the games that matter.

    MUTATION: make the resolver return False instead of delegating and this fails.
    """

    monkeypatch.delenv("CARNOT_ARC_WM_CHANGE_GATE_HIDDEN_STATE", raising=False)
    monkeypatch.setenv("CARNOT_ARC_WM_CHANGE_GATE", "1")
    assert world_model_change_gate_hidden_state_enabled() is True


def test_hidden_state_flag_can_be_isolated_from_req6011_in_both_directions(monkeypatch):
    """The explicit override is what lets a follow-up separate the two branches without a
    code edit. Both directions are asserted: an override that only worked one way would
    leave half the matrix unreachable."""

    monkeypatch.setenv("CARNOT_ARC_WM_CHANGE_GATE", "1")
    monkeypatch.setenv("CARNOT_ARC_WM_CHANGE_GATE_HIDDEN_STATE", "0")
    assert world_model_change_gate_hidden_state_enabled() is False

    monkeypatch.setenv("CARNOT_ARC_WM_CHANGE_GATE", "0")
    monkeypatch.setenv("CARNOT_ARC_WM_CHANGE_GATE_HIDDEN_STATE", "1")
    assert world_model_change_gate_hidden_state_enabled() is True


# ---------------------------------------------------------------------------
# The coverage hole itself.
# ---------------------------------------------------------------------------


def test_incumbent_consistency_is_literally_blind_to_an_invented_write():
    """The MECHANISM of the hole, not merely its effect.

    `heldout_change_consistency` for the honest engine and for the same engine plus an
    invented write must be EXACTLY equal -- not close, equal. Equality is what proves the
    quantity cannot see the invention at all, which is a stronger and more durable claim
    than "the attack passed too" (that could be threshold luck).

    MUTATION: this test documents the incumbent's behaviour, so it fails if
    `score_change_weighted_consistency` is ever changed to a symmetric metric -- at which
    point this file's premise is obsolete and the failure is the correct alarm.
    """

    honest = _select(_perfect)
    attack = _select(_perfect_plus_invention)
    assert attack.heldout_change_consistency == honest.heldout_change_consistency
    assert attack.trust_pass == honest.trust_pass


def test_select_trusted_world_model_attaches_a_populated_change_gate_record():
    """MUTATION: drop the `change_gate=` argument from `_candidate_score` and this fails.

    The record must be populated in EVERY arm including a control, because a control whose
    field is empty gives the four-arm matrix nothing to compare against.
    """

    score = _select(_perfect)
    assert score.change_gate, "change_gate record must be populated, not empty"
    for key in ("passed", "reason", "change_fidelity", "n_changing", "spurious_changed_cells"):
        assert key in score.change_gate


def test_the_symmetric_gate_rejects_the_invented_write_the_incumbent_admits():
    """THE HOLE, CLOSED. Both halves are asserted: the incumbent admits, the repair rejects.

    Asserting only the rejection would pass even if the incumbent had also rejected, in
    which case the repair would have closed nothing.

    MUTATION: revert `spurious_changed_cells`/union scoring to recall-only and this fails.
    """

    attack = _select(_perfect_plus_invention)
    assert attack.trust_pass is True, "premise: the incumbent admits this engine"
    assert attack.change_gate_pass is False
    assert attack.change_gate["spurious_changed_cells"] > 0


def test_the_symmetric_gate_still_admits_the_honest_engine():
    """MUST-NOT-FIRE. A gate that rejects everything is not an improvement over a gate that
    admits identity engines.

    MUTATION: raise WORLD_MODEL_CHANGE_FIDELITY_THRESHOLD above the honest engine's
    fidelity and this fails -- which is the point: it pins the pass region as non-empty.
    """

    honest = _select(_perfect)
    assert honest.change_gate_pass is True
    assert honest.change_gate["reason"] == "passed"


def test_identity_engine_is_rejected_even_though_the_corpus_is_noop_heavy():
    """The GAP-WM-TRUST-GATE origin shape: `accuracy` reads high because most transitions
    are no-ops, and the gate must not be fooled by it.

    MUTATION: gate on `legacy_accuracy` instead of `change_fidelity` and this fails.
    """

    ident = _select(_identity)
    assert ident.change_gate["legacy_accuracy_would_pass"] is True, (
        "premise: the LEGACY gate admits the identity engine on this corpus -- if this "
        "premise ever stops holding, the corpus no longer reproduces the incident"
    )
    assert ident.change_gate_pass is False
    assert ident.change_gate["reason"] == "degenerate_engine_no_correct_changed_cells"


def test_change_gate_is_scored_on_the_same_heldout_split_as_trust_pass():
    """Guards the double-split bug that rejected even a perfect engine in an earlier
    harness: a caller that pre-split and passed the tail in would have it split AGAIN.

    The check is structural -- the gate's transition count must equal the held-out size
    (one third), not the whole corpus -- so it fails if the computation is ever moved back
    out to a caller that does not own the split.

    MUTATION: score the change gate on `transitions` instead of `heldout` and this fails.
    """

    corpus = _noop_heavy_corpus()
    score = select_trusted_world_model(
        corpus, [_candidate(_perfect)], hidden_state=True
    ).selected_score
    n_heldout = max(1, int(round(len(corpus) / 3.0)))
    assert score.change_gate["n_transitions"] == n_heldout
    assert score.change_gate["n_transitions"] < len(corpus)


def test_change_gate_pass_defaults_to_false_on_a_missing_record():
    """A decision that could not be computed is not evidence of trustworthiness. Defaulting
    the other way would make a plumbing failure look like a pass.

    MUTATION: change the property's default to True and this fails.
    """

    from carnot.agentic.arc_world_model_trust_energy import CandidateScore

    score = CandidateScore(
        candidate=_candidate(_perfect),
        prefix_accuracy=1.0,
        heldout_accuracy=1.0,
        trust_energy=0.0,
        baseline_clears=True,
        heldout_best=True,
    )
    assert score.change_gate == {}
    assert score.change_gate_pass is False


# ---------------------------------------------------------------------------
# The UNION metric, isolated from the no-op channel.
# ---------------------------------------------------------------------------
#
# WHY THESE TWO TESTS EXIST. A mutation proof (reverting `union = m | wrote` to `union = m`,
# i.e. back to recall-only) did NOT fail the suite. The reason is instructive rather than
# cosmetic: the earlier attack engine invents a cell on EVERY transition, so it trips the
# no-op hallucination channel and is rejected regardless of what the union metric does. The
# test asserting "the symmetric gate rejects the invented write" was therefore passing for a
# reason other than the one in its name -- a pass that could not have failed for the stated
# cause. These two tests remove the no-op channel from the corpus entirely (no true no-op
# exists, the real re86 shape) so that ONLY the union metric can decide.


def _noop_free_corpus(n: int = 18) -> list[Transition]:
    """Every transition changes -- so `n_noop == 0` and the no-op channel cannot fire.

    This is not a contrived shape: it is re86's real held-out split (40 of 40 changing),
    where a spurious writer clears the entire gate at fidelity 0.919 for exactly this
    reason. See the exp6013 artifact's residual-escape witness.
    """

    return [
        Transition(_grid((1, i % 4), i), 1, None, _grid((1, (i % 4) + 1), i + 1), 0, 0)
        for i in range(n)
    ]


def _inventing(k: int):
    """`_perfect` plus `k` invented cells per transition, in a row reality never touches."""

    def engine(grid, action, data):
        g = np.asarray(_perfect(grid, action, data)).copy()
        g.reshape(-1)[np.arange(k) + (W * 3)] = 9
        return g

    return engine


def test_union_metric_alone_rejects_a_heavy_invented_writer_with_no_noop_channel():
    """The union metric doing the work UNAIDED. The corpus has no no-op, so the channel
    that catches the every-transition attack elsewhere is structurally unavailable here.

    MUTATION: revert `union = m | wrote` to `union = m` (recall-only) and this fails. That
    mutation survived the rest of the suite, which is why this test exists.
    """

    corpus = _noop_free_corpus()
    vr = WorldModelVerifier(corpus).score(_inventing(4))
    assert vr.noop_channel_measurable is False, "premise: the no-op channel cannot fire here"
    decision = change_gate_decision(vr, enabled=True)
    assert decision["passed"] is False
    assert decision["reason"] == "change_fidelity_below_threshold", (
        "and it must be the FIDELITY that rejects, not some other channel -- otherwise "
        "this test would again be passing for a reason other than its name"
    )


def test_union_metric_alone_does_NOT_catch_a_light_invented_writer_known_limitation():
    """PINS A KNOWN LIMITATION rather than hiding it.

    Union fidelity is an ABSOLUTE threshold while a spurious write is a small RELATIVE
    decrement: one invented cell against three genuinely-changed ones scores 0.75, comfortably
    above 0.5. So on a corpus with no no-ops, a LIGHT invented writer is admitted. That is
    precisely the re86 escape measured in the exp6013 artifact (fidelity 0.881-0.919, all
    three seeds, both mask settings), and it is the reason that artifact reports the hole as
    majority-closed-and-explained rather than closed.

    This test asserts the CURRENT behaviour deliberately. If a future change strengthens the
    gate so this engine is rejected, this test fails -- and that failure is the correct
    signal to re-measure re86 and update the artifact's residual claim, not to delete the
    test. Perverse-looking assertions of known weaknesses are how a limitation stays visible
    instead of quietly becoming folklore.
    """

    corpus = _noop_free_corpus()
    vr = WorldModelVerifier(corpus).score(_inventing(1))
    assert vr.noop_channel_measurable is False
    assert vr.invented_changed_cells > 0, "it really does invent"
    decision = change_gate_decision(vr, enabled=True)
    assert decision["passed"] is True, (
        "KNOWN LIMITATION: a light invented writer clears the absolute fidelity threshold "
        "when the true change set is large and no no-op exists to catch it"
    )
    assert decision["noop_ok_is_vacuous"] is True, (
        "and the channel that WOULD have caught it is flagged as unmeasurable rather than "
        "reporting the value that means 'clean'"
    )


# ---------------------------------------------------------------------------
# The two new diagnostics (recorded, deliberately NOT gated on).
# ---------------------------------------------------------------------------


def test_noop_channel_reports_unmeasurable_rather_than_clean_when_there_are_no_noops():
    """`noop_hallucination_rate` returns 0.0 when `n_noop == 0`, so the value meaning "this
    engine invents nothing" is ALSO the value meaning "this could not be measured". On the
    real re86 corpus all 40 held-out transitions change, and a spurious writer clears the
    whole gate at fidelity 0.919 precisely because the channel that would have caught it
    cannot fire. This asserts the two meanings are now distinguishable.

    MUTATION: delete `noop_channel_measurable` / `noop_ok_is_vacuous` and this fails.
    """

    all_changing = [
        Transition(_grid((1, i % 4), i), 1, None, _grid((1, (i % 4) + 1), i + 1), 0, 0)
        for i in range(12)
    ]
    vr = WorldModelVerifier(all_changing).score(_perfect_plus_invention)
    assert vr.n_noop == 0
    assert vr.noop_hallucination_rate == 0.0, "the misleading value is still what is reported"
    assert vr.noop_channel_measurable is False, "but it is now flagged as unmeasurable"
    decision = change_gate_decision(vr, enabled=True)
    assert decision["noop_ok"] is True
    assert decision["noop_ok_is_vacuous"] is True, (
        "the noop_ok verdict passed because there was nothing to test, not because the "
        "engine is clean -- a consumer that conflates these reads a false pass"
    )


def test_invented_cells_are_distinguished_from_ordinary_prediction_error():
    """`spurious_changed_cells` is `wrote & ~correct`, which CONFLATES a cell invented out
    of nothing with a genuinely-changed cell predicted at the wrong value -- ordinary error
    that every imperfect-but-useful engine has. Only `invented_changed_cells` (`wrote & ~m`)
    isolates invention. This is the distinction that must exist before anyone calibrates a
    threshold on it.

    The engine here invents NOTHING and is merely wrong about a real change, so the two
    counters must DISAGREE. If they agreed, the new field would be a redundant alias.

    MUTATION: define `invented_changed_cells` as `wrote & ~correct` and this fails.
    """

    def wrong_value_on_a_real_change(grid, action, data):
        g = np.asarray(_perfect(grid, action, data)).copy()
        where = np.argwhere(g == 3)
        if len(where):
            r, c = where[0]
            g[r, c] = 7  # right cell, wrong value: error, not invention
        return g

    vr = WorldModelVerifier(_noop_heavy_corpus()).score(wrong_value_on_a_real_change)
    assert vr.spurious_changed_cells > 0, "it does write cells that are not correct"
    assert vr.invented_changed_cells == 0, (
        "but it never touches a cell reality left alone -- so this is prediction error, "
        "not invention, and the two quantities must not be the same number"
    )


def test_invented_cells_fire_on_a_genuine_invention():
    """The must-fire half of the pair above. Without it, `invented_changed_cells == 0` in
    the previous test could be satisfied by a field that is always zero -- a dead channel.

    MUTATION: hardcode `invented_changed_cells = 0` and this fails while the test above
    still passes, which is exactly why both directions are needed.
    """

    vr = WorldModelVerifier(_noop_heavy_corpus()).score(_perfect_plus_invention)
    assert vr.invented_changed_cells > 0
    assert vr.invented_change_rate > 0.0


# ---------------------------------------------------------------------------
# BRANCH COVERAGE -- the defect class itself.
# ---------------------------------------------------------------------------


def test_both_admission_branches_consult_the_change_gate():
    """The ORIGINAL defect was not a wrong gate, it was a gate wired into one of two
    branches. `change_gate_decision` had exactly one call site, in the `else` arm, so the 11
    hidden-state games -- every 0.08-wall game -- were ungated. Nothing in the test suite
    could see that, because every existing test exercised the library function or the `else`
    arm; a unit test of a gate cannot detect that a branch never calls it.

    So this test asserts over the AST of `_induce_and_plan` that BOTH arms of the
    `self.short in HIDDEN_STATE_GAME_IDS` branch reference the gate. It is deliberately
    structural: that is the only level at which "this code path was forgotten" is visible.

    MUTATION: delete the change-gate block from the hidden-state arm (i.e. restore the
    pre-REQ-6013 code) and this fails, while every other test in this file still passes --
    which is precisely the blind spot it exists to cover.
    """

    import ast
    import inspect
    import textwrap

    from carnot.agentic.arc_competition_agent import E3AgentPolicy

    # dedent: a method's source is indented inside its class, which `ast.parse` rejects.
    tree = ast.parse(textwrap.dedent(inspect.getsource(E3AgentPolicy._induce_and_plan)))
    branches = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If) and "HIDDEN_STATE_GAME_IDS" in ast.unparse(node.test)
    ]
    assert len(branches) == 1, (
        f"expected exactly one hidden-state dispatch, found {len(branches)}; if the "
        "dispatch was refactored this test must be updated deliberately, not deleted"
    )
    branch = branches[0]
    hidden_arm = "\n".join(ast.unparse(s) for s in branch.body)
    plain_arm = "\n".join(ast.unparse(s) for s in branch.orelse)

    assert "change_gate" in plain_arm, (
        "premise: the non-hidden-state arm already consulted the change gate (REQ-6011)"
    )
    assert "change_gate" in hidden_arm, (
        "the hidden-state arm must consult the change gate -- this is the REQ-6013 repair"
    )
    assert "world_model_change_gate_hidden_state_enabled" in hidden_arm, (
        "and it must do so behind its own resolvable flag, not unconditionally"
    )

    # PRESENCE OF THE TEXT IS NOT ENOUGH. A mutation that replaced the guard with
    # `if False:` left every assertion above satisfied -- the gate code was still there,
    # just unreachable -- and the whole suite stayed green. So the guard's CONDITION is
    # checked too: it must be a live expression, never a constant, and it must be the one
    # that decides the rejection.
    guards = [
        node
        for node in ast.walk(ast.Module(body=branch.body, type_ignores=[]))
        if isinstance(node, ast.If)
        and "change_gate_pass" in "\n".join(ast.unparse(s) for s in node.body)
    ]
    assert guards, "no conditional in the hidden-state arm actually acts on change_gate_pass"
    for guard in guards:
        assert not isinstance(guard.test, ast.Constant), (
            f"the change-gate guard is the constant `{ast.unparse(guard.test)}`, so the "
            "gate is dead code: present in the source, never consulted at runtime"
        )


# ---------------------------------------------------------------------------
# THE ORIGIN INCIDENT, against the REAL file on disk.
# ---------------------------------------------------------------------------


def test_the_real_ondisk_lp85_engine_is_rejected_by_the_hidden_state_change_gate():
    """NON-HERMETIC ON PURPOSE -- this is the origin-incident test.

    `results/arc_e3/lp85/world_model.py` is one of the two real engines GAP-WM-TRUST-GATE
    was written about; it mutates only on `action == 6 and grid[py, px] == 9` and was
    reported to the operator as "the good model". It is loaded through the production
    `e3.load_engine` and routed through the production `select_trusted_world_model`.

    The rejection is additionally required to be DISCRIMINATING rather than vacuous: over a
    held-out split containing at least one genuinely changing transition. Rejecting an
    engine on a corpus where nothing could pass is a pass that could not have failed, and
    the sibling ft09 corpus is exactly that case (0 of 40 held-out transitions change), so
    this distinction is not academic.

    MUTATION: revert the hidden-state `change_gate` wiring and this fails.
    """

    # NOT skipped when the corpus is unavailable -- FAILED, loudly. A guard that has never
    # been run against its own origin incident is not a guard, so "we could not load it"
    # must be a red test, not a quiet green one.
    assert "error" not in _ORIGIN_CORPUS, (
        f"could not load the origin-incident engine/corpus: {_ORIGIN_CORPUS.get('error')}"
    )
    engine = _ORIGIN_CORPUS["engine"]
    transitions = _ORIGIN_CORPUS["transitions"]

    score = select_trusted_world_model(
        list(transitions), [_candidate(engine, "ondisk_lp85")], hidden_state=True
    ).selected_score

    assert score.change_gate["n_changing"] > 0, (
        "the rejection must be decided over a non-empty changing population, or it is "
        "vacuous -- see the VACUOUS set in the exp6013 artifact for the ft09 case"
    )
    assert score.change_gate_pass is False
    assert score.change_gate["reason"] == "degenerate_engine_no_correct_changed_cells"
    assert score.change_gate["correct_changed_cells"] == 0

"""REQ-ARC-WMTE-6071 / SCENARIO-ARC-WMTE-6071-1..8: exact-label inert-action deferral.

Every test here fails against the pre-6071 code, either because the symbol it names does not
exist (`arc_inert_label_memory`, `inert_label_memory`, `_inert_label_keep_indices`,
`_select_untested_index`, `inert_label_defer_diagnostics`) or because the behaviour it asserts
is not there (a node whose known-inert rows are still popped first).

Deliberately NOT an arcade run: these pin the DECISION RULE and the SAFETY LADDER, which are
what a future edit is most likely to loosen. The arcade-level result -- whether deferring
actually buys progress -- is carried by the A/B artifact, not by the unit suite, because a unit
test that could answer it would have to be the experiment.
"""

from __future__ import annotations

import pytest

from carnot.agentic.arc_competition_agent import StepwiseExplorer
from carnot.agentic.arc_inert_label_memory import (
    InertLabelMemory,
    coerce_inert_label_memory,
    label_key,
)


# ---------------------------------------------------------------------------------------
# The memory itself
# ---------------------------------------------------------------------------------------


def test_label_key_separates_click_coordinates_and_keeps_keyboard_keyable():
    """SCENARIO-ARC-WMTE-6071-1. The key is the LITERAL action, which is the whole difference
    from the retired structural-signature pruner: two clicks on different pixels are different
    keys even when they land on identical-looking objects."""

    assert label_key(6, {"x": 3, "y": 4}) == (6, 3, 4)
    assert label_key(6, {"x": 3, "y": 5}) != label_key(6, {"x": 3, "y": 4})
    assert label_key(2, None) == (2, None, None)
    # A click whose payload has no coordinates still keys on the action id rather than
    # vanishing -- an unkeyable row would be silently un-deferrable forever.
    assert label_key(6, {}) == (6, None, None)
    # Undecodable action -> no key at all, so the memory can neither record nor defer it.
    assert label_key("not-an-action", None) is None


def test_one_inert_observation_makes_the_same_literal_label_deferrable():
    """SCENARIO-ARC-WMTE-6071-2. min_observations=1 is the measured operating point (98.4%
    precision on the roster traces); this pins that the default really is 1, so raising it
    silently is a test failure rather than an unnoticed recall loss."""

    mem = InertLabelMemory()
    assert mem.min_observations == 1
    assert not mem.is_deferrable_row({"action": 6, "data": {"x": 1, "y": 1}})
    mem.observe(6, {"x": 1, "y": 1}, unchanged=True)
    assert mem.is_deferrable_row({"action": 6, "data": {"x": 1, "y": 1}})
    # ...and ONLY that label. The neighbouring pixel is untouched.
    assert not mem.is_deferrable_row({"action": 6, "data": {"x": 2, "y": 1}})


def test_a_label_that_ever_changed_the_frame_is_never_deferrable_again():
    """SCENARIO-ARC-WMTE-6071-3, safety brake 1. Order-independent: whether the effect is seen
    before or after the inert observations, the label is retired from the memory."""

    after = InertLabelMemory()
    after.observe(6, {"x": 1, "y": 1}, unchanged=True)
    after.observe(6, {"x": 1, "y": 1}, unchanged=True)
    assert after.is_deferrable_row({"action": 6, "data": {"x": 1, "y": 1}})
    after.observe(6, {"x": 1, "y": 1}, unchanged=False)
    assert not after.is_deferrable_row({"action": 6, "data": {"x": 1, "y": 1}})

    before = InertLabelMemory()
    before.observe(6, {"x": 1, "y": 1}, unchanged=False)
    for _ in range(50):
        before.observe(6, {"x": 1, "y": 1}, unchanged=True)
    assert not before.is_deferrable_row({"action": 6, "data": {"x": 1, "y": 1}})


def test_a_label_that_ever_leveled_up_is_sacred_even_if_it_is_also_recorded_inert():
    """SCENARIO-ARC-WMTE-6071-4, safety brake 2, along the path `observe` really produces."""

    mem = InertLabelMemory()
    mem.observe(2, None, unchanged=False, leveled_up=True)
    for _ in range(100):
        mem.observe(2, None, unchanged=True)
    assert not mem.is_deferrable_row({"action": 2, "data": None})
    assert mem.stats()["labels_sacred_leveled"] == 1


def test_the_levelup_veto_holds_on_its_own_without_the_frame_change_veto():
    """SCENARIO-ARC-WMTE-6071-4b. Brake 2 tested INDEPENDENTLY of brake 1.

    THIS TEST EXISTS BECAUSE THE VETO WAS DECORATIVE WHEN FIRST WRITTEN. `observe` counts a
    level-up in BOTH the `leveled` and `effective` slots, so along every path `observe` can
    produce, brake 1 already covers brake 2 -- and deleting the `n_leveled > 0` line left the
    whole suite GREEN. A veto that can be deleted without a test noticing is one refactor away
    from being removed as dead code, and the thing it protects is a winning action. So the
    counters are set directly here to reach `leveled > 0, effective == 0`, the one state
    `observe` cannot reach, which is the only state in which brake 2 is load-bearing.
    """

    mem = InertLabelMemory()
    mem.set_counts_for_test(2, None, inert=100, effective=0, leveled=1)
    assert not mem.is_deferrable_row({"action": 2, "data": None})
    assert mem.stats()["labels_sacred_leveled"] == 1
    # Control: identical counters WITHOUT the level-up are deferrable, so the assertion above
    # is about the veto and not about some unrelated guard.
    mem.set_counts_for_test(2, None, inert=100, effective=0, leveled=0)
    assert mem.is_deferrable_row({"action": 2, "data": None})


def test_coercion_matches_the_sibling_pruners_contract():
    assert coerce_inert_label_memory(None) is None
    assert coerce_inert_label_memory(False) is None
    assert isinstance(coerce_inert_label_memory(True), InertLabelMemory)
    inst = InertLabelMemory(min_observations=4)
    assert coerce_inert_label_memory(inst) is inst
    assert coerce_inert_label_memory("garbage") is None


# ---------------------------------------------------------------------------------------
# The live-path decision rule
# ---------------------------------------------------------------------------------------


def _rows(*coords):
    return [{"action": 6, "data": {"x": x, "y": y}} for x, y in coords]


def test_flag_off_leaves_the_pop_rule_untouched():
    """SCENARIO-ARC-WMTE-6071-5. The lever ships INERT. `_inert_label_keep_indices` must return
    None on the shipped default so `_pop_untested_inner` runs exactly the pre-6071 path -- this
    is what makes the A/B a comparison rather than a rewrite."""

    ex = StepwiseExplorer("g")
    assert ex.inert_label_memory is None
    assert ex.inert_label_defer_enabled is False
    node = {"untested": _rows((1, 1), (2, 2), (3, 3))}
    assert ex._inert_label_keep_indices(node["untested"]) is None
    # And the pop is still the head of the list (tier barrier inactive: no click frame seen).
    assert ex._pop_untested_inner(node) == {"action": 6, "data": {"x": 1, "y": 1}}


def test_a_known_inert_head_row_is_skipped_in_favour_of_an_unknown_one():
    """SCENARIO-ARC-WMTE-6071-6. THE CORE BEHAVIOUR, and the one that fails against the pre-6071
    code: the head row is known-inert, so the pop takes the first row the memory has no
    complaint about instead. The skipped row is still IN the list."""

    ex = StepwiseExplorer("g", inert_label_memory=True)
    ex.inert_label_memory.observe(6, {"x": 1, "y": 1}, unchanged=True)
    node = {"untested": _rows((1, 1), (2, 2), (3, 3))}
    popped = ex._pop_untested_inner(node)
    assert popped == {"action": 6, "data": {"x": 2, "y": 2}}
    assert {"action": 6, "data": {"x": 1, "y": 1}} in node["untested"]
    assert len(node["untested"]) == 2
    diag = ex.inert_label_defer_diagnostics()
    assert diag["deferred_pops"] == 1
    assert diag["rows_deferred"] == 1


def test_deferral_is_never_a_drop_when_every_remaining_row_is_known_inert():
    """SCENARIO-ARC-WMTE-6071-7, safety brake 3 -- the fail-open, and the single most important
    difference from the RETIRED signature pruner. When the memory has an opinion about EVERY
    row it abstains completely: the node pops on today's schedule, its `untested` length falls
    exactly as it would have, and therefore `_node_has_open_tier`, the frontier and the
    navigation budget are untouched. The retired pruner dropped instead, and its own post-mortem
    attributes +12.0% states_expanded to that."""

    ex = StepwiseExplorer("g", inert_label_memory=True)
    for x, y in ((1, 1), (2, 2), (3, 3)):
        ex.inert_label_memory.observe(6, {"x": x, "y": y}, unchanged=True)
    node = {"untested": _rows((1, 1), (2, 2), (3, 3))}
    assert ex._inert_label_keep_indices(node["untested"]) is None
    assert ex.inert_label_defer_diagnostics()["abstained"] == 1  # that consultation abstained
    popped = ex._pop_untested_inner(node)
    assert popped == {"action": 6, "data": {"x": 1, "y": 1}}  # unchanged from today
    assert len(node["untested"]) == 2  # the row was SPENT, not silently dropped
    # 2, not 1: the pop consults the memory again and abstains again. Counting consultations
    # rather than pops is deliberate -- `abstained` is how the artifact tells "the lever had no
    # jurisdiction here" apart from "the lever was never asked".
    assert ex.inert_label_defer_diagnostics()["abstained"] == 2
    assert ex.inert_label_defer_diagnostics()["deferred_pops"] == 0


def test_the_memory_never_empties_a_single_row_node():
    """A one-row node is never a choice, so the memory must not be consulted at all -- consulting
    it and finding the only row deferrable is the shape that turns a defer into a drop."""

    ex = StepwiseExplorer("g", inert_label_memory=True)
    ex.inert_label_memory.observe(6, {"x": 1, "y": 1}, unchanged=True)
    node = {"untested": _rows((1, 1))}
    assert ex._inert_label_keep_indices(node["untested"]) is None
    assert ex._pop_untested_inner(node) == {"action": 6, "data": {"x": 1, "y": 1}}
    assert node["untested"] == []


def test_deferral_respects_the_frontier_tier_barrier_rather_than_overriding_it():
    """SCENARIO-ARC-WMTE-6071-8. The tier barrier decides WHETHER a row is admitted; the memory
    only reorders among admitted rows. Here the only non-deferrable row sits above the active
    tier, so the filtered draw finds nothing admitted and the pop falls through to the
    unfiltered barrier draw -- the memory yields, it does not widen the barrier."""

    ex = StepwiseExplorer("g", inert_label_memory=True)
    ex.tier_exhaustion_enabled = True
    ex.tier_click_vocab_only = False
    ex._active_tier = 0
    ex.inert_label_memory.observe(6, {"x": 1, "y": 1}, unchanged=True)
    node = {
        "untested": [
            {"action": 6, "data": {"x": 1, "y": 1}, "tier": 0},  # known inert, ADMITTED
            {"action": 6, "data": {"x": 9, "y": 9}, "tier": 3},  # unknown, DEFERRED BY TIER
        ]
    }
    popped = ex._pop_untested_inner(node)
    assert popped["data"] == {"x": 1, "y": 1}
    assert ex.inert_label_defer_diagnostics()["deferred_pops"] == 0


def test_diagnostics_distinguish_a_dead_channel_from_a_real_null():
    """A zero prune count has several causes and the artifact must be able to tell them apart --
    the exact conflation that made a previous lever's null uninterpretable."""

    off = StepwiseExplorer("g").inert_label_defer_diagnostics()
    assert off["enabled"] is False and off["observe_calls"] == 0

    ex = StepwiseExplorer("g", inert_label_memory=True)
    on = ex.inert_label_defer_diagnostics()
    assert on["enabled"] is True and on["flag_resolved"] is True
    assert on["observe_calls"] == 0  # dead channel until _ingest runs
    assert on["labels_tracked"] == 0
    assert on["verifier_is_oracle"] is False


def test_env_flag_enables_without_a_kwarg_and_is_off_by_default(monkeypatch):
    monkeypatch.delenv("CARNOT_ARC_INERT_LABEL_DEFER", raising=False)
    assert StepwiseExplorer("g").inert_label_memory is None
    monkeypatch.setenv("CARNOT_ARC_INERT_LABEL_DEFER", "1")
    assert StepwiseExplorer("g").inert_label_memory is not None
    monkeypatch.setenv("CARNOT_ARC_INERT_LABEL_DEFER", "0")
    assert StepwiseExplorer("g").inert_label_memory is None


def test_explicit_kwarg_beats_the_env_override(monkeypatch):
    monkeypatch.setenv("CARNOT_ARC_INERT_LABEL_DEFER", "1")
    assert StepwiseExplorer("g", inert_label_memory=False).inert_label_memory is None


@pytest.mark.parametrize("k", [1, 2, 4])
def test_evidence_floor_is_honoured(k):
    mem = InertLabelMemory(min_observations=k)
    for i in range(k):
        assert not mem.is_deferrable_row({"action": 3, "data": None})
        mem.observe(3, None, unchanged=True)
    assert mem.is_deferrable_row({"action": 3, "data": None})

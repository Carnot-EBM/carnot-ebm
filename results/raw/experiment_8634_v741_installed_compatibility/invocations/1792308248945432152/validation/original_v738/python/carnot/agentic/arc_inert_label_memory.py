"""Exact-label inert-action memory: DEFER a probe the agent has already watched do nothing.

REQ-ARC-WMTE-6071. Default OFF (``SUBMITTED_INERT_LABEL_DEFER_ENABLED = False`` in
``arc_competition_agent.py``); env override ``CARNOT_ARC_INERT_LABEL_DEFER=1``.

THE MEASURED FACT THIS EXISTS FOR
---------------------------------
The 2026-08-02 roster action census
(``results/arc_explorer_renavigation_20260802/arc_explorer_renavigation.json``) decomposed
the scored explorer's action budget into named classes and found the LARGEST AVOIDABLE
class is not re-navigation at all -- it is ``expansion.probe_was_inert_frame_unchanged``:
1,148 of 6,000 actions (19.1%) at budget 240, and 9,208 of 50,000 (18.4%) when the same
25-game roster is re-run at budget 2000. An "inert" action is one after which the RAW
frame is byte-identical to the frame before it: the agent spent an action and not one
pixel moved.

WHY THE SAME ACTION KEEPS GETTING PAID FOR
------------------------------------------
``StepwiseExplorer`` builds a node's ``untested`` candidate list ONCE, per node, from the
node's own frame. Two nodes that show the same object therefore both offer the same click
coordinate, and each pays for it separately -- the search's "have I tried this?" bookkeeping
is per-node, so it cannot see that this exact action already did nothing somewhere else.
Measured on the budget-2000 roster traces, the single most-repeated inert label was probed
356 times in one run (s5i5), 245 times in another (ar25), and 148-164 times in three more
(sk48, sc25, sb26).

Concretely, over 43,533 probe actions on the 25-game roster: an exact label that had ALREADY
been observed inert at least once, and never observed doing anything, was inert again
**6,601 times out of 6,710 predictions -- 98.4% precision, 71.7% recall** over all inert
probes. On the seven games carrying most of the inert mass the per-game precision is 0.996
to 1.000 (ft09, lp85, sc25, sk48, sb26, su15, ar25).

WHAT THIS IS NOT: THE RETIRED SIGNATURE PRUNER
----------------------------------------------
``arc_inert_click_pruner.InertClickSigPruner`` is a RETIRED-NEGATIVE lever
(``results/outer_loop_inert_click_pruner_shipped_config_ab_20260726.json``: zero new wins on
75 matched pairs, lost ft09 on 2 of 3 seeds, states_expanded +12.0% pooled / +37.9% in the
non-HUD stratum; recommendation "DO NOT FLIP ... retire the lever in its current RAW-GRID-
inertness form"). This module is not a re-run of it, and the two differ on the three axes
that its own post-mortem named:

* **KEY.** The retired pruner keys on a STRUCTURAL GENERALIZATION -- Reki's ``(color,
  pixel_count, is_rect, twin_count)`` blob signature -- so evidence about one blob suppresses
  clicks on every look-alike blob anywhere on the board. That is why it needed a 4-observation
  evidence floor plus a 0.9 specificity threshold and still mis-fired. This module keys on the
  LITERAL ``(action_id, x, y)`` the agent will actually send. It generalizes across STATES
  (which is where the waste is) and across nothing else.
* **CONSEQUENCE.** The retired pruner DROPS rows from ``node["untested"]``, which shortens the
  node's list, changes ``_node_has_open_tier``, retires the node from the frontier earlier and
  buys more navigation -- the mechanism behind its measured +12% search cost. This module
  never drops anything. It only changes WHICH row a node pops next, and only while that node
  still has at least one non-deferred row. A deferred row stays in ``untested``, keeps the
  node frontier-eligible on exactly the same schedule, and is popped normally the moment it is
  all that is left. The action space is never narrowed; the ORDER of spending inside a node is.
* **CHANNEL.** The retired pruner's learning channel was ``awaiting["previous_frame"]``, which
  is populated only when one of nine unrelated optional components happens to be attached (the
  dead-channel defect documented at length in ``arc_competition_agent._ingest``). This module
  learns from ``_last_unmasked_hash``/``_unmasked_hash``, which ``_ingest`` maintains
  UNCONDITIONALLY, so its evidence cannot be silenced by turning some other component off.

WHY "UNMASKED" AND NOT NODE IDENTITY
------------------------------------
Node identity is the HUD-masked hash: it answers "is this a new node to the search". Inertness
is the strictly cheaper question "did anything on the screen change at all". They differ
exactly where they matter -- an action that only ticks a HUD counter changes the raw frame
(so it is NOT inert; something happened, the agent simply cannot use it to tell states apart),
and an action that returns the board to a state already in the graph changes the raw frame too
(that is ``expansion.probe_revisited_known_state``, a class the census marks NOT avoidable
because the transition edge it buys is real information). Keying on the raw hash keeps this
memory pointed at literally-nothing-happened and nothing else.

SAFETY
------
Three independent brakes, in the order they bind:

1. **A label that has EVER done anything is never deferrable.** One observed frame change
   retires the label from the memory permanently.
2. **A label that has EVER produced a level-up is SACRED** -- a separate hard veto kept even
   though ``observe`` also counts a level-up as an effect, mirroring both sibling pruners,
   because a level-up is categorically too valuable to leave to one counter. The two vetoes
   are exercised INDEPENDENTLY by ``tests/python/test_arc_inert_label_memory.py`` (deleting
   either one alone fails the suite), because a veto whose deletion leaves the suite green is
   a decorative veto.
3. **Deferral is never a drop.** When every remaining row at a node is deferrable the memory
   abstains entirely and the node pops exactly as it does today.

Checked against the traces the mechanism was designed on: of the **24 level-up actions** the
25-game budget-2000 roster produced, **0** would have been deferred.

``verifier_is_oracle: False`` -- a frequency memory fit from the agent's own observed
transitions, never the executable oracle that defines correctness. It reads only frames the
live agent already has and requires no adapter, no game source and no generator, so it is as
available on a hidden game as on a public one.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional


def label_key(action_id: Any, data: Any) -> Optional[tuple[int, Optional[int], Optional[int]]]:
    """The literal action the agent will send, as a hashable key.

    ``(action_id, x, y)`` for a click carrying coordinates, ``(action_id, None, None)`` for
    everything else (keyboard moves, and clicks whose payload has no usable coordinates).
    Returns None for a label that cannot be decoded to an integer action id -- the memory
    never records, and therefore never defers, what it cannot key.
    """

    try:
        aid = int(action_id)
    except (TypeError, ValueError):
        return None
    if isinstance(data, Mapping) and "x" in data and "y" in data:
        try:
            return (aid, int(data["x"]), int(data["y"]))
        except (TypeError, ValueError):
            return (aid, None, None)
    return (aid, None, None)


class InertLabelMemory:
    """Per-run tally of which exact action labels have been watched doing nothing.

    Lifetime is one ``StepwiseExplorer`` -- one episode on one game. Nothing is persisted
    and nothing is loaded, so on a hidden game the memory starts empty and is filled purely
    by that game's own observed transitions.
    """

    verifier_is_oracle = False

    def __init__(self, *, min_observations: int = 1) -> None:
        # min_observations=1 is the measured operating point: one prior inert observation of
        # the SAME literal label already predicts inertness at 98.4% precision on the roster
        # traces, and raising it to 2 buys 0.9pp of precision for 2.8pp of recall. It is a
        # constructor parameter rather than a constant so the A/B can move it without editing
        # the live path.
        self.min_observations = max(1, int(min_observations))
        # key -> [n_inert, n_effective, n_leveled]
        self._tally: dict[tuple[int, Optional[int], Optional[int]], list[int]] = {}
        self.observed = 0
        self.observe_errors = 0

    # -- learning -----------------------------------------------------------------------
    def observe(
        self,
        action_id: Any,
        data: Any,
        *,
        unchanged: bool,
        leveled_up: bool = False,
    ) -> None:
        """Record one realized transition.

        ``unchanged`` is the caller's raw-frame verdict (byte-identical before/after). It is
        passed in rather than computed here so this class never has to know how the live
        agent hashes a frame, and so the caller can use the UNCONDITIONALLY-maintained
        unmasked hash rather than a node frame that may not exist.
        """

        key = label_key(action_id, data)
        if key is None:
            self.observe_errors += 1
            return
        row = self._tally.setdefault(key, [0, 0, 0])
        if leveled_up:
            row[2] += 1
            # A level-up is also, trivially, an effect. Counting it in BOTH slots means safety
            # brakes 1 and 2 each independently retire the label along the path `observe`
            # actually produces. The brakes are still tested SEPARATELY (via a hand-set tally
            # with a level-up and no effect), because a brake that only ever fires behind
            # another brake is one refactor away from being deleted as dead code.
            row[1] += 1
        elif unchanged:
            row[0] += 1
        else:
            row[1] += 1
        self.observed += 1

    def set_counts_for_test(
        self,
        action_id: Any,
        data: Any,
        *,
        inert: int = 0,
        effective: int = 0,
        leveled: int = 0,
    ) -> None:
        """Set a label's raw counters directly. Never called from the live path.

        Exists so the two safety vetoes can be exercised INDEPENDENTLY. ``observe`` deliberately
        counts a level-up in both the ``leveled`` and ``effective`` slots, so no sequence of
        ``observe`` calls can produce ``leveled > 0, effective == 0`` -- which means a test built
        only from ``observe`` cannot tell whether the level-up veto is load-bearing or dead. It
        was in fact dead-by-that-test when first written: deleting the veto left the suite green,
        which is the "pattern is narrower than its concept / untested pattern" class CLAUDE.md's
        Test-Run Record Integrity Discipline names.
        """

        key = label_key(action_id, data)
        if key is None:
            return
        self._tally[key] = [int(inert), int(effective), int(leveled)]

    # -- prediction ---------------------------------------------------------------------
    def is_deferrable_key(self, key: Optional[tuple[int, Optional[int], Optional[int]]]) -> bool:
        if key is None:
            return False
        row = self._tally.get(key)
        if row is None:
            return False
        n_inert, n_effective, n_leveled = row
        if n_leveled > 0:
            return False  # SACRED: this label has completed a level before
        if n_effective > 0:
            return False  # it has done something at least once -- never defer it again
        return n_inert >= self.min_observations

    def is_deferrable_row(self, row: Any) -> bool:
        """``node["untested"]`` row form: ``{"action": int, "data": dict|None, ...}``."""

        if not isinstance(row, Mapping):
            return False
        return self.is_deferrable_key(label_key(row.get("action"), row.get("data")))

    # -- reporting ----------------------------------------------------------------------
    def stats(self) -> dict[str, Any]:
        deferrable = sum(1 for key in self._tally if self.is_deferrable_key(key))
        return {
            "observed": int(self.observed),
            "observe_errors": int(self.observe_errors),
            "labels_tracked": len(self._tally),
            "labels_deferrable": int(deferrable),
            "labels_sacred_leveled": sum(1 for row in self._tally.values() if row[2] > 0),
            "min_observations": int(self.min_observations),
            "verifier_is_oracle": False,
        }


def coerce_inert_label_memory(value: Any) -> Optional[InertLabelMemory]:
    """Constructor coercion matching ``coerce_inert_click_pruner``'s shape: ``None``/``False``
    -> no memory; an already-constructed instance -> passthrough; ``True`` -> a default
    instance."""

    if value is None or value is False:
        return None
    if isinstance(value, InertLabelMemory):
        return value
    if value is True:
        return InertLabelMemory()
    return None

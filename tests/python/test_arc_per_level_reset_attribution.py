"""Tests for PER-LEVEL reset attribution in the ARC leaderboard eval harness.

WHY THESE SPECIFIC TESTS
========================
The defect being guarded is not "a counter is wrong". It is a family of ACCOUNTING defects that
each read as a plausible measurement, which is why this project has shipped several of them:

  - a whole-run total substituted for a per-level attribution. `n_resets` is a single number for
    the entire episode, but the scorer's per-level denominator is the SPAN between two consecutive
    level-ups (arc_agi/scorecard.py:479 differences cumulative charged counts). A total cannot be
    apportioned across spans after the fact, so a harness that records only the total makes every
    per-level efficiency number individually unknowable -- while still looking instrumented.
  - a number in an UNNAMED unit. Three units coexist here (offline_actions excludes resets, frames
    includes them, gateway_charged is what the scorer bills) and conflating two of them has already
    flipped a conclusion in this project: one cell's 2775-action gap is ~2936 frames.
  - CUMULATIVE checkpoints reported as if they were spans, which inflates every level after the
    first and is invisible unless the spans are summed back against the run total.
  - a DEAD CHANNEL that reads as a measured value rather than as an error. `eff_gateway` defaulting
    to 0.0 made `efficiency_optimism_vs_gateway` report the full value of `eff` -- "offline
    accounting is 100% optimistic", the most alarming reading available -- precisely when nothing
    had been measured. The sibling case was a `getattr(env, "baseline_actions")` against a field
    that lives on `env.info`, which made a broken probe look like a clean null.
  - a PROJECTION that silently drops the fields needed to interpret the rest. The nav diagnostics
    were narrowed from 24 keys to 2, discarding exactly the reset-classifying counters, which is
    why a downstream analysis had to re-run the agent live rather than read persisted rows.

Every test below is MUTATION-PROVED: the docstring names the mutation it catches, and each was
applied and confirmed to turn the test red before being reverted. Nine mutations are covered.

Spec: REQ-ARC-WMTE-5985 -- SCENARIO-ARC-WMTE-5985-ATTRIBUTION-IS-PER-SEGMENT-NOT-WHOLE-RUN,
SCENARIO-ARC-WMTE-5985-EVERY-SPAN-CARRIES-ALL-THREE-UNITS,
SCENARIO-ARC-WMTE-5985-SPANS-RECONCILE-AGAINST-THE-RUN-TOTAL,
SCENARIO-ARC-WMTE-5985-A-LEVEL-JUMP-CLOSES-ONE-SPAN-PER-LEVEL,
SCENARIO-ARC-WMTE-5985-ATTRIBUTION-POPULATES-ON-THE-CRASH-PATH,
SCENARIO-ARC-WMTE-5985-AN-ABSENT-CHANNEL-IS-NOT-A-MEASURED-ZERO,
SCENARIO-ARC-WMTE-5985-THE-NAV-PROJECTION-KEEPS-THE-RESET-CLASSIFIERS,
SCENARIO-ARC-WMTE-5985-A-BROKEN-GATEWAY-SCORE-IS-NONE-NOT-MAXIMAL-OPTIMISM, and
SCENARIO-ARC-WMTE-5985-THE-INSTRUMENTATION-IS-A-PROVEN-PURE-ADDITION (proved out-of-band by a
byte-identical trajectory fingerprint before/after the diff, not by a unit test -- a test cannot
observe the pre-change build of the code it imports).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _load():
    """Import the eval harness by path.

    `scripts/` is not an installed package. The harness itself inserts `python/` on sys.path at
    import time, so its own `carnot.*` / `arcengine` imports resolve; we only have to make
    `scripts/` importable for the module name to bind.
    """

    if str(REPO / "scripts") not in sys.path:
        sys.path.insert(0, str(REPO / "scripts"))
    spec = importlib.util.spec_from_file_location(
        "arc_leaderboard_eval_under_test", REPO / "scripts" / "arc_leaderboard_eval.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


LB = _load()


# =========================================================================================
# Fakes. These drive the REAL run_game loop -- the point is to exercise the accumulators in
# situ, not to re-implement them in the test. A frame only needs `.levels_completed`
# (arc_agi3_live_adapter._levels_completed reads exactly that attribute).
# =========================================================================================


class _Frame:
    def __init__(self, levels_completed: int):
        self.levels_completed = int(levels_completed)
        self.state = "NOT_FINISHED"
        self.available_actions: list = []


class _FakeEnv:
    """Replays a SCRIPTED level schedule so a span's contents are known exactly.

    `schedule` maps a 0-based move index to the levels_completed value observed AFTER that move.
    Between scripted entries the level is carried forward, so spans have deterministic lengths.
    """

    def __init__(self, schedule: dict[int, int], baselines=None, die_at: int | None = None):
        self.schedule = dict(schedule)
        self.i = -1
        self.level = 0
        self.die_at = die_at
        self.info = type("I", (), {"baseline_actions": list(baselines or [])})()

    def _advance(self):
        self.i += 1
        if self.i in self.schedule:
            self.level = int(self.schedule[self.i])
        if self.die_at is not None and self.i >= self.die_at:
            return None  # the crash path: run_game appends None to frames and breaks
        return _Frame(self.level)

    def reset(self):
        return self._advance()

    def step(self, action, data=None):
        return self._advance()


class _FakePolicy:
    """Emits a scripted move sequence. `kinds` entries are "RESET" or an action id string."""

    def __init__(self, kinds: list[str], explorer=None):
        self.kinds = list(kinds)
        self.n = 0
        if explorer is not None:
            self.explorer = explorer

    def is_done(self, frames, latest):
        return self.n >= len(self.kinds)

    def next_move(self, frames, latest):
        kind = self.kinds[self.n]
        self.n += 1
        return kind, ({} if kind == "RESET" else None)


def _run(kinds, schedule, *, baselines=None, die_at=None, explorer=None, budget=None):
    """Drive the real run_game against the fakes."""

    env = _FakeEnv(schedule, baselines=baselines, die_at=die_at)
    arcade = type(
        "A",
        (),
        {"open_scorecard": lambda self: "sc", "make": lambda self, g, scorecard_id=None: env},
    )()
    orig = LB.kit.offline_arcade
    LB.kit.offline_arcade = lambda: arcade
    try:
        return LB.run_game(
            "fake", _FakePolicy(kinds, explorer=explorer), budget=budget or len(kinds) + 2
        )
    finally:
        LB.kit.offline_arcade = orig


# =========================================================================================
# THE HEADLINE DEFECT: a whole-run total is not an attribution.
# =========================================================================================


def test_resets_are_attributed_per_span_not_as_a_whole_run_total():
    """MUTATION CAUGHT (1/9): replacing per-span attribution with the whole-run `n_resets`.

    The schedule puts a level-up at move 3 and another at move 9, with resets deliberately
    UNEVENLY distributed: 2 resets before the first level-up and 3 more before the second. A
    harness that reports the whole-run total (5) per level cannot produce the asymmetric pair
    (2, 3), so this assertion is exactly what a total-substituted mutation fails.
    """

    # moves 0..3 -> span 1 (2 resets, 2 actions); moves 4..9 -> span 2 (3 resets, 3 actions)
    kinds = ["RESET", "RESET", "1", "1", "RESET", "RESET", "RESET", "1", "1", "1"]
    r = _run(kinds, {3: 1, 9: 2})
    a = r["level_reset_attribution"]

    assert a["segment_resets"] == [2, 3], a["segment_resets"]
    # The whole-run total is a DIFFERENT number from either span. If it happened to equal a span
    # the mutation would be undetectable here, so assert it is distinct.
    assert a["run_total_resets"] == 5
    assert a["run_total_resets"] not in a["segment_resets"]


def test_every_span_carries_all_three_units_and_gateway_charged_includes_resets():
    """MUTATION CAUGHT (2/9): computing `gateway_charged` from offline actions alone.

    The three units must be simultaneously present AND numerically distinct whenever a span
    contains a reset -- that is the whole reason the units are tracked separately. A mutation
    that sets gateway_charged = offline_actions (dropping the reset charge the live gateway
    bills at scorecard.py:701-704) collapses the distinction and fails here.
    """

    kinds = ["RESET", "RESET", "1", "1", "RESET", "RESET", "RESET", "1", "1", "1"]
    a = _run(kinds, {3: 1, 9: 2})["level_reset_attribution"]

    for seg in a["segments"]:
        for unit in ("offline_actions", "resets", "frames", "gateway_charged"):
            assert unit in seg, unit
        assert seg["gateway_charged"] == seg["offline_actions"] + seg["resets"]
        # frames INCLUDE resets, offline_actions EXCLUDE them -> frames is the larger of the two.
        assert seg["frames"] == seg["offline_actions"] + seg["resets"]
        assert seg["gateway_charged"] > seg["offline_actions"]  # this span really has resets

    assert a["segment_offline_actions"] == [2, 3]
    assert a["segment_gateway_charged"] == [4, 6]
    assert a["segment_frames"] == [4, 6]
    assert a["unit_definitions"]["gateway_charged"].startswith("offline_actions + resets")


def test_spans_are_differences_not_cumulative_checkpoints():
    """MUTATION CAUGHT (3/9): emitting cumulative checkpoints as if they were spans.

    `level_up_charged` is deliberately CUMULATIVE (4, 10 here); the spans must be its successive
    differences (4, 6). Summing spans + tail back to the run total is the check that catches the
    confusion: cumulative values over-sum, and the reconciliation flag goes false.
    """

    kinds = ["RESET", "RESET", "1", "1", "RESET", "RESET", "RESET", "1", "1", "1"]
    r = _run(kinds, {3: 1, 9: 2})
    a = r["level_reset_attribution"]

    assert r["level_up_charged"] == [4, 10]  # cumulative, unchanged
    assert a["segment_gateway_charged"] == [4, 6]  # differenced
    total = sum(a["segment_gateway_charged"]) + a["tail_gateway_charged"]
    assert total == a["run_total_gateway_charged"]
    assert a["reconciles"] is True
    assert a["discrepancies"] == []


def test_reconciliation_actually_fails_when_the_spans_do_not_add_up():
    """MUTATION CAUGHT (4/9): hardcoding `reconciles = True` (a forced gate).

    A gate that cannot fail is not a gate. Feeding deliberately inconsistent totals must produce
    reconciles=False AND a non-empty, named discrepancy list -- if the flag were forced, or the
    tolerance widened, this stays green and the whole cross-check is decoration.
    """

    segs = [{"offline_actions": 2, "resets": 2, "frames": 4, "level_completed": 1}]
    # A frame_sequence consistent with `segs`: seed frame, then 2 resets + 2 actions, level-up last.
    fs = [
        {"move": {"kind": "1"}, "levels_completed": 0},
        {"move": {"kind": "RESET"}, "levels_completed": 0},
        {"move": {"kind": "RESET"}, "levels_completed": 0},
        {"move": {"kind": "1"}, "levels_completed": 1},
    ]
    bad = LB._build_level_reset_attribution(
        segments=segs,
        open_tail={"offline_actions": 0, "resets": 0, "frames": 0},
        frame_sequence=fs,
        total_offline_actions=99,  # deliberately wrong
        total_resets=2,
    )
    assert bad["reconciles"] is False
    assert bad["discrepancies"], "an inconsistent accounting must NAME its discrepancy"
    assert any("offline_actions" in d["check"] for d in bad["discrepancies"])
    # And the clean case must be genuinely clean, so the test above is not passing vacuously.
    good = LB._build_level_reset_attribution(
        segments=segs,
        open_tail={"offline_actions": 0, "resets": 0, "frames": 0},
        frame_sequence=fs,
        total_offline_actions=2,
        total_resets=2,
    )
    assert good["reconciles"] is True and good["discrepancies"] == []
    # The channel-2 disagreement is ITSELF a detectable discrepancy class, not just the totals:
    # a frame_sequence that does not match the accumulators must be named too.
    mismatched = LB._build_level_reset_attribution(
        segments=segs,
        open_tail={"offline_actions": 0, "resets": 0, "frames": 0},
        frame_sequence=[],  # channel 2 sees no spans at all
        total_offline_actions=2,
        total_resets=2,
    )
    assert mismatched["reconciles"] is False
    assert any("channel2" in d["check"] for d in mismatched["discrepancies"])


def test_a_multi_level_jump_closes_one_span_per_level_with_the_cost_on_the_first():
    """MUTATION CAUGHT (5/9): appending ONE span for a multi-level jump.

    A jump from level 0 to level 2 in a single frame must close TWO spans -- the first carrying the
    whole cost, the second zero -- because the gateway's `actions_by_level` appends one entry per
    observed change and the scorer charges the jumped levels off the tail. A mutation that appends
    one span per JUMP rather than per LEVEL leaves n_segments at 1 and silently mis-attributes.
    """

    kinds = ["RESET", "1", "1", "1"]
    a = _run(kinds, {2: 2})["level_reset_attribution"]  # 0 -> 2 in one frame

    assert a["n_segments"] == 2, a["n_segments"]
    assert [s["level_completed"] for s in a["segments"]] == [1, 2]
    assert a["segment_gateway_charged"] == [3, 0]  # whole cost first, then zero
    assert a["segment_resets"] == [1, 0]
    assert a["reconciles"] is True


def test_attribution_populates_on_the_crash_path_and_never_substitutes_none_for_zero():
    """MUTATION CAUGHT (6/9): guarding the attribution behind `if levels:` (so it is None).

    A run that dies mid-episode, or never levels up at all, must still emit REAL numbers -- the
    open span becomes the tail. An absent key that a reader takes for a measured zero is the
    defect; so is a None where a zero is meant. Both branches are asserted.
    """

    # (a) env returns None mid-run -> run_game breaks out of the loop
    r = _run(["RESET", "1", "1", "1", "1"], {}, die_at=3)
    a = r["level_reset_attribution"]
    assert a is not None and isinstance(a, dict)
    assert a["n_segments"] == 0  # never levelled up
    assert a["tail_frames"] > 0, "the tail must hold the work that WAS done before the crash"
    assert a["reconciles"] is True

    # (b) no level-up at all: zeros must be present ints, not None, not missing
    a2 = _run(["1", "1"], {})["level_reset_attribution"]
    for key in ("tail_resets", "resets_in_completed_segments", "resets_in_tail"):
        assert a2[key] is not None, key
        assert isinstance(a2[key], int), key
    assert a2["resets_in_completed_segments"] == 0  # a MEASURED zero
    assert a2["segment_resets"] == []  # no spans -> empty list, not None
    assert a2["discrepancies"] == []  # empty LIST when clean, never None


def test_the_independent_frame_sequence_channel_agrees_with_the_in_loop_accumulators():
    """MUTATION CAUGHT (7/9): dropping the second accounting channel (or making it vacuous).

    Two independent derivations that must agree is the only way an off-by-one in a counting loop
    announces itself. The in-loop accumulators and the frame_sequence re-derivation are computed
    from different sources; their spans must match unit-for-unit.
    """

    kinds = ["RESET", "RESET", "1", "1", "RESET", "RESET", "RESET", "1", "1", "1"]
    a = _run(kinds, {3: 1, 9: 2})["level_reset_attribution"]
    ch2 = a["channel2_frame_sequence_derived"]

    assert ch2["n_segments"] == a["n_segments"] == 2
    for seg_acc, seg_ch2 in zip(a["segments"], ch2["segments"]):
        for unit in ("offline_actions", "resets", "frames", "gateway_charged"):
            assert seg_acc[unit] == seg_ch2[unit], (unit, seg_acc, seg_ch2)
    # and the standalone channel-2 function must be usable on a persisted row alone
    standalone = LB.segment_attribution_from_frame_sequence(
        [
            {"move": {"kind": "RESET"}, "levels_completed": 0},
            {"move": {"kind": "1"}, "levels_completed": 0},
            {"move": {"kind": "1"}, "levels_completed": 1},
        ]
    )
    assert standalone["n_segments"] == 1
    assert standalone["segments"][0]["offline_actions"] == 2
    assert standalone["segments"][0]["resets"] == 1
    assert standalone["segments"][0]["gateway_charged"] == 3


# =========================================================================================
# THE PROJECTION DEFECTS: dropped classifiers, and a dead channel that reads as a value.
# =========================================================================================


class _Explorer:
    def __init__(self, payload=None, raise_it=False):
        self.payload = payload
        self.raise_it = raise_it

    def navigation_diagnostics(self):
        if self.raise_it:
            raise RuntimeError("diagnostics blew up")
        return dict(self.payload or {})


def test_the_nav_projection_keeps_the_reset_classifying_counters():
    """MUTATION CAUGHT (8/9): narrowing nav diagnostics back to the two-key projection.

    The two legacy keys alone cannot classify a single reset. `reset_replay_fallbacks` and
    `navigation_attempts` plus the exact/partial/similarity split are what separate an
    irreducible navigation reset from a fixable one; dropping them is why a prior analysis had to
    re-run the agent live instead of reading rows. The legacy keys must ALSO survive.
    """

    payload = {
        "navigation_attempts": 87,
        "reset_replay_fallbacks": 87,
        "exact_shortest_path_hits": 14,
        "partial_forward_walk_hits": 0,
        "similarity_forward_walk_hits": 0,
        "forward_edges_recorded": 51,
        "reset_replay_steps": 5,
        "forward_walk_hit_rate": 0.0,
    }
    nav = LB._navigation_diagnostics(_FakePolicy([], explorer=_Explorer(payload)))

    for key in (
        "navigation_attempts",
        "reset_replay_fallbacks",
        "exact_shortest_path_hits",
        "partial_forward_walk_hits",
        "similarity_forward_walk_hits",
        "forward_edges_recorded",
    ):
        assert key in nav, f"the projection dropped {key}, the reset cannot be classified"
    assert nav["reset_replay_fallbacks"] == 87
    # legacy keys retained AND correctly typed (run_game reads them positionally)
    assert nav["reset_replay_steps"] == 5 and isinstance(nav["reset_replay_steps"], int)
    assert nav["forward_walk_hit_rate"] == 0.0 and isinstance(nav["forward_walk_hit_rate"], float)
    assert nav["instrumented"] is True and nav["uninstrumented_reason"] is None


def test_an_absent_nav_channel_is_flagged_rather_than_reported_as_a_measured_zero():
    """MUTATION CAUGHT (9/9): returning bare `{reset_replay_steps: 0, hit_rate: 0.0}` on failure.

    A 0 that means "never measured" is indistinguishable from a 0 that means "measured, no resets"
    -- and this project already shipped that exact defect as a dead
    `getattr(env, "baseline_actions")` read against a field on `env.info`, where the dead channel
    read as a clean null. All three failure paths must be flagged, with a machine-readable reason,
    while KEEPING the legacy zeros so no existing consumer breaks.
    """

    no_explorer = LB._navigation_diagnostics(object())
    lacks_method = LB._navigation_diagnostics(_FakePolicy([], explorer=object()))
    raised = LB._navigation_diagnostics(_FakePolicy([], explorer=_Explorer(raise_it=True)))
    wrong_type = LB._navigation_diagnostics(_FakePolicy([], explorer=_Explorer(payload=None)))

    for nav in (no_explorer, lacks_method, raised):
        assert nav["instrumented"] is False
        assert isinstance(nav["uninstrumented_reason"], str) and nav["uninstrumented_reason"]
        # legacy compatibility: the zeros are still there, they are just no longer ambiguous
        assert nav["reset_replay_steps"] == 0 and nav["forward_walk_hit_rate"] == 0.0
    assert "raised" in raised["uninstrumented_reason"]
    # an empty-but-real dict is MEASURED, so it must NOT be flagged uninstrumented
    assert wrong_type["instrumented"] is True

    # A genuine all-zero measurement must read as instrumented, or the flag is useless.
    real_zero = LB._navigation_diagnostics(
        _FakePolicy([], explorer=_Explorer({"reset_replay_fallbacks": 0, "reset_replay_steps": 0}))
    )
    assert real_zero["instrumented"] is True
    assert real_zero["reset_replay_fallbacks"] == 0


def test_an_unmeasurable_gateway_score_is_none_not_the_full_value_of_eff():
    """MUTATION CAUGHT (bonus): restoring `eff_gateway = 0.0` as the default.

    `efficiency_optimism_vs_gateway` is `eff - eff_gateway`. With eff_gateway defaulting to 0.0, a
    game whose env exposes no human baselines silently reports the FULL value of eff as optimism --
    "offline accounting is 100% optimistic", the most alarming reading the field can express,
    emitted exactly when nothing was measured. It must be None with a stated reason instead.
    """

    # no baselines -> the gateway score is unmeasurable
    r = _run(["RESET", "1", "1", "1"], {2: 1}, baselines=None)
    assert r["efficiency_gateway_charged"] is None
    assert r["efficiency_optimism_vs_gateway"] is None
    assert r["efficiency_gateway_charged_error"] == "no_baseline_actions_exposed_by_env"

    # WITH baselines the channel must actually produce a number, so the check above is not
    # passing because the feature is simply dead everywhere.
    r2 = _run(["RESET", "1", "1", "1"], {2: 1}, baselines=[2, 5, 9])
    assert isinstance(r2["efficiency_gateway_charged"], float)
    assert r2["efficiency_gateway_charged_error"] is None
    assert r2["efficiency_optimism_vs_gateway"] is not None
    # resets are charged, so the gateway score can only be <= the reset-free offline score
    assert r2["efficiency_gateway_charged"] <= r2["efficiency"]
    assert r2["efficiency_optimism_vs_gateway"] >= 0.0

"""Drift guards for the SCORED-PATH lever harness (``scripts/arc_scored_path_lever_harness.py``).

WHY A TEST FOR A HARNESS. The harness declares its own ARMS dict, independent of
``experiment_5836_frontier_discipline_ab``'s. That means it can drift the same way exp5836's arms
drifted: arm B2 there pinned 3 of 7 gated flags, so the 2026-07-25 flip silently turned the HUD A/B's
own CONTROL into the HUD TREATMENT, and published numbers changed meaning with nobody editing the
arm. Two tests in ``test_arc_frontier_discipline.py`` now make that impossible for exp5836. These
tests do the same job for this harness, plus the two things unique to it: that the control arm still
equals the LIVE configuration, and that the nav-pruner cell classifier cannot report a non-firing
lever as a null.

No network, no GPU, no game environment: everything here is pure declaration + classifier logic.

Spec refs: REQ-ARC-WMTE-5970, SCENARIO-ARC-WMTE-5970-FIRE-COUNTERS.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

_HARNESS = Path(__file__).resolve().parents[2] / "scripts" / "arc_scored_path_lever_harness.py"


def _harness() -> ModuleType:
    """Load the harness by path. It lives in scripts/ (not an installed package) and is loaded the
    same way the sibling exp5836 harness tests load theirs."""

    spec = importlib.util.spec_from_file_location("arc_scored_path_lever_harness", _HARNESS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_harness_arm_pins_every_gated_flag():
    """An arm that pins only a SUBSET inherits module defaults for the rest, so a later flag flip
    redefines that arm underneath already-published numbers. This is the exact defect that made
    exp5836's arm B2 the HUD treatment; it must not be reintroduced in a second harness."""

    m = _harness()
    keys = set(m.SHIPPED_2026_07_25)
    assert len(keys) == 8, f"expected the eight gated flags, got {sorted(keys)}"
    assert "hazard_move_pruner" in keys, "the nav-side pruner must be pinned like every other lever"
    for name, kwargs in m.ARMS.items():
        assert set(kwargs) == keys, f"arm {name} does not pin exactly the gated set"


@pytest.mark.memory_watchdog_skip
def test_the_control_arm_is_the_live_configuration():
    """Pinning keeps an ARM stable; only this check tells you whether the arm named "S" is still the
    LIVE config. If a SUBMITTED_* flag is flipped and nobody updates the harness, "S" silently stops
    being the control -- and every delta measured against it becomes a two-lever delta.

    MARKED `memory_watchdog_skip` because this is the only test here that imports
    `carnot.agentic.arc_competition_agent` (via `assert_shipped_dict_matches_module_globals`, which
    must read the LIVE `SUBMITTED_*` globals -- reading them from anywhere else would defeat the
    check's entire purpose). That import allocates ~580MB one time, which trips the conftest's 500MB
    per-test leak threshold at TEARDOWN. It is a module-import cost, not a leak, and the marker is
    the project's established escape hatch for exactly this case (used by ~30 other agent-importing
    tests). Without it this test reports an ERROR alongside its PASS, which is precisely the kind of
    always-red signal that trains a reader to ignore the suite.
    """

    m = _harness()
    parity = m.assert_shipped_dict_matches_module_globals()
    assert parity["pinned_vs_live_drift"] == {}, (
        f"the harness's pinned control no longer matches the live SUBMITTED_* globals: "
        f"{parity['pinned_vs_live_drift']} -- update SHIPPED_2026_07_25 (and re-read any verdict "
        f"measured against arm S) before using it as a control"
    )
    assert m.ARMS["S"] == m.SHIPPED_2026_07_25


def test_each_single_lever_arm_differs_from_the_control_in_exactly_its_own_lever():
    """A single-lever arm that differs in two levers measures the pair, not the lever. Computed
    rather than eyeballed, because eyeballing is what let the B2 contamination through."""

    m = _harness()
    expected = {
        "S_minus_frontier": set(m._FRONTIER_KEYS),
        "S_minus_hud": set(m._HUD_KEYS),
        "S_plus_hazard": {"hazard_move_pruner"},
        "S_minus_both": set(m._FRONTIER_KEYS) | set(m._HUD_KEYS),
    }
    for arm, want in expected.items():
        got = {k for k in m.SHIPPED_2026_07_25 if m.ARMS[arm][k] != m.ARMS["S"][k]}
        # frontier_gradient is already False in the shipped config, so turning it "off" is a no-op
        # difference -- compare against the flags that actually CHANGE.
        assert got <= want, f"arm {arm} also differs in {sorted(got - want)}"
        assert got, f"arm {arm} is identical to the control -- it measures nothing"


def test_the_budget_default_is_the_shipped_agents_own_action_cap():
    """The SHIPPED agent is capped at MAX_ACTIONS=400 per game (the CarnotAgent adapter in
    ``arc_competition_agent``; the framework loop is ``while not done and action_counter <=
    MAX_ACTIONS``), so 400 is the condition the CURRENT SUBMISSION runs under and that is why it is
    the default here.

    CORRECTED 2026-07-26. This test's previous docstring, and the harness prose it guarded, said 400
    was what "the eval allows". It is not: the comment directly above that constant states that the
    real bound is the eval's wall-clock budget (<=12h across all games) and that MAX_ACTIONS is an
    INTENDED OVERRIDE POINT (Playback sets it to 1e6). The misreading is load-bearing because lever
    orderings REVERSE with the budget -- at 2000 the shipped configuration is the best of four arms
    (median 12 wins in the convention-transfer battery), at 400 it wins 3-4 of 25 -- so a
    recommendation derived from one budget alone is not supported.

    Asserted against the value a caller who passes no ``--budget`` actually receives, not against a
    separate constant that could drift away from the parser.
    """

    import re

    m = _harness()
    source = open(m.__file__, encoding="utf-8").read()
    match = re.search(r'add_argument\(\s*"--budget",\s*type=int,\s*default=(\d+)\s*\)', source)
    assert match, "could not find the --budget argument definition"
    assert int(match.group(1)) == 400, (
        f"harness default budget is {match.group(1)}, expected 400 (the shipped agent's own cap)"
    )
    # And the run artifact must record whether the budget it used matched that cap, plus what the
    # cap MEANS, so neither a budget-400 nor a budget-2000 run can be read as the other.
    assert "budget_matches_scored_cap" in source
    assert "scored_agent_max_actions" in source
    assert "budget_semantics" in source
    # The prose must not reassert the misreading.
    assert "more play than the eval allows" not in source, (
        "the harness prose again claims 400 is an eval-imposed bound; it is a self-imposed "
        "per-game loop guard and a documented override point"
    )
    assert "self-imposed" in source and "INTENDED OVERRIDE POINT" in source


def test_hazard_verdict_separates_a_non_firing_lever_from_a_real_null():
    """THE CLASSIFIER THAT MATTERS. ``rows_pruned == 0`` has four structurally different causes and
    only two of them are evidence about the lever. Reporting a dead observe channel, a game with no
    nav actions, or an unfitted hypothesis class as "the lever does not help" is the exp5836
    dead-channel failure -- a clean, zero-error, byte-identical NULL that was pure harness artifact.
    """

    m = _harness()
    v = m._hazard_verdict

    assert v({}, {}) == "ERROR", "unreadable diagnostics prove nothing"
    assert v({"error": "boom"}, {}) == "ERROR"
    assert v({"enabled": False}, {}) == "LEVER_OFF", "the control arm makes no claim either way"

    # observe_calls == 0 with the lever ENABLED is a WIRING BUG, and must never read as a null.
    dead = {
        "enabled": True,
        "observe_calls": 0,
        "observed_nav_transitions": 0,
        "model_fitted": False,
        "rows_pruned": 0,
    }
    assert v(dead, {}) == "UNINTERPRETABLE_NO_OBSERVE"

    # The hook ran but nothing was accepted -> a 100%-click game. No jurisdiction.
    no_nav = {
        "enabled": True,
        "observe_calls": 393,
        "observed_nav_transitions": 0,
        "model_fitted": False,
        "rows_pruned": 0,
    }
    assert v(no_nav, {}) == "UNINTERPRETABLE_NO_NAV"

    # Nav transitions seen, but no rung passed the evidence/trust/specificity gate.
    unfitted = {
        "enabled": True,
        "observe_calls": 388,
        "observed_nav_transitions": 383,
        "n_deaths": 1,
        "model_fitted": False,
        "rows_pruned": 0,
    }
    assert v(unfitted, {}) == "UNINTERPRETABLE_NOT_FITTED"

    # Fitted and predicted nothing lethal -- the ONLY reportable null.
    fired_null = {
        "enabled": True,
        "observe_calls": 388,
        "observed_nav_transitions": 383,
        "n_deaths": 6,
        "model_fitted": True,
        "rows_pruned": 0,
    }
    assert v(fired_null, {}) == "FIRED_NO_PRUNE"

    fired = dict(fired_null, rows_pruned=88)
    assert v(fired, {}) == "FIRED_AND_PRUNED"


def test_the_four_uninterpretable_verdicts_are_not_counted_as_fired():
    """`lever3_fired` and `lever3_interpretable` must agree with the verdict string, so a downstream
    aggregation cannot accidentally pool a non-firing cell into a lever's denominator."""

    m = _harness()
    firing = {"FIRED_NO_PRUNE", "FIRED_AND_PRUNED"}
    uninterpretable = {
        "UNINTERPRETABLE_NO_OBSERVE",
        "UNINTERPRETABLE_NO_NAV",
        "UNINTERPRETABLE_NOT_FITTED",
        "ERROR",
    }
    # LEVER_OFF is interpretable (it is the control) but not firing -- assert that asymmetry, since
    # collapsing it into either bucket would either lose the control or inflate the fire count.
    assert firing & uninterpretable == set()
    assert "LEVER_OFF" not in firing and "LEVER_OFF" not in uninterpretable
    source = open(m.__file__, encoding="utf-8").read()
    for token in firing | uninterpretable | {"LEVER_OFF"}:
        assert token in source, f"{token} must be produced by the harness"


def test_hud_lever_fire_predicate_is_not_anti_correlated_with_its_own_lever():
    """THE FIRE COUNTER THAT WAS BROKEN AND UNTESTED. Real recorded detector output, not a hand-set
    flag.

    DEFECT REPRODUCED (measured 2026-07-26): the first predicate ANDed in
    ``hud_shipped_mask_digest``, i.e. it required the ALREADY-SHIPPED ``auto_hud_mask`` classifier to
    have produced a mask before the REPAIRED detector's mask could count as a difference. But the
    entire reason REQ-ARC-WMTE-5960 exists is that the shipped classifier resolves None on r11l and
    tn36 -- the ONLY two games where the repaired detector resolves a mask it does not. So the
    counter was anti-correlated with the lever it measures and read 0 in all 430 cells of the first
    scored-path run, while the lever demonstrably fired (r11l: mask None -> 64 cells,
    states_expanded 319 -> 41; tn36: None -> 61 cells, 49 -> 17, deterministic on 3/3 seeds).

    Lever 1 and lever 3 both had verdict tests; the one lever that was broken was the one with none.
    """

    m = _harness()
    # Copied verbatim from recorded rows of
    # results/outer_loop_scored_path_lever_ab_llm_on_20260726.json (fields the predicate reads).
    r11l = {
        "hud_mask_resolved": True,
        "hud_mask_source": "edge_bar_detector_req5960_stage2_confirmed",
        "hud_mask_cell_count": 64,
        "hud_mask_digest": "fcbba0b6818499b6",
        "hud_shipped_mask_cell_count": 0,
        "hud_shipped_mask_digest": None,
    }
    tn36 = dict(r11l, hud_mask_cell_count=61, hud_mask_digest="791a436c692cdbf8")
    assert m.hud_lever_fired(r11l) is True, (
        "a mask APPEARING where the shipped config had none is the lever's strongest firing"
    )
    assert m.hud_lever_fired(tn36) is True

    # NEGATIVE CONTROLS. The fix must not collapse into "resolved == fired".
    # lf52: the shipped classifier already resolves this exact mask, so the repair adds nothing.
    lf52 = {
        "hud_mask_resolved": True,
        "hud_mask_source": "status_bar_classifier_req5583_no_repair_added_cell",
        "hud_mask_cell_count": 64,
        "hud_mask_digest": "e92122951bcd64e7",
        "hud_shipped_mask_cell_count": 64,
        "hud_shipped_mask_digest": "e92122951bcd64e7",
    }
    assert m.hud_lever_fired(lf52) is False
    # HUD trio off / no bar detected: nothing resolved.
    assert (
        m.hud_lever_fired(
            {
                "hud_mask_resolved": False,
                "hud_mask_source": "unresolved_no_bar_detected",
                "hud_mask_digest": None,
                "hud_shipped_mask_digest": None,
            }
        )
        is False
    )
    # Unreadable diagnostics prove nothing in either direction.
    assert m.hud_lever_fired({"error": "AttributeError:boom"}) is False
    assert m.hud_lever_fired({}) is False
    assert m.hud_lever_fired(None) is False
    # A same-COUNT but different mask must still fire: the 2026-07-25 gate compared counts and
    # therefore read a same-size different mask as "no change".
    same_count_diff_mask = dict(lf52, hud_mask_digest="0000000000000000")
    assert m.hud_lever_fired(same_count_diff_mask) is True


# ---------------------------------------------------------------------------------------------
# THE POPULATION GAP (2026-07-26). The test above proves the PREDICATE is right. It cannot prove
# the FIELDS THE PREDICATE READS ARE EVER POPULATED, because it hand-writes them as dict literals
# -- which is precisely how a sibling wiring test passed against a completely dead observe channel
# (0 of 122 graph nodes carried `previous_frame`; the test injected the field itself). Everything
# below therefore takes its field VALUES from a live `StepwiseExplorer.hud_mask_diagnostics()` call
# reached through the SCORED policy's own attribute chain. Nothing here writes a hud_mask_* key.
# ---------------------------------------------------------------------------------------------


def _live_scored_explorer(grids):
    """A REAL `E3AgentPolicy` -- the scored policy -- with real frames ingested through its own
    `.explorer`, returning `(explorer, diagnostics)`.

    THE CHAIN IS THE POINT. The suspected cause of the None-valued diagnostics was that the harness
    reads them off an object reachable on the `CarnotAgentPolicy` path but not on the
    `E3AgentPolicy` path. Constructing the SCORED policy and going through `policy.explorer` here is
    what makes that a measured fact rather than an assumption: if the attribute or the method were
    missing on this path, this helper raises instead of quietly yielding Nones.

    No game environment, no network, no GPU: `environment_files/` is gitignored, so a test that
    needed a real game could not run in CI and would have to be skipped -- and a skipped test is an
    invisible failure. Frames are synthetic; every hud_mask_* VALUE is computed by the real detector.
    """

    import numpy as np

    from carnot.agentic.arc_competition_agent import E3AgentPolicy, StepwiseExplorer

    class _Frame:
        """Minimal arcengine-frame stand-in: `grid_of` reads only `.frame`."""

        def __init__(self, grid) -> None:
            self.frame = grid
            self.state = "NOT_FINISHED"
            self.levels_completed = 0

    # A game id that is deliberately NOT one of the 25 public games, so the test cannot accidentally
    # depend on a per-game GameAdapter or on anything under the gitignored environment_files/.
    policy = E3AgentPolicy("zzz_synthetic_not_a_public_game")
    explorer = getattr(policy, "explorer", None)
    assert isinstance(explorer, StepwiseExplorer), (
        "the SCORED policy must expose its explorer as `.explorer`; if this ever stops being true "
        "the harness's `ex = policy.explorer` reads None and every hud_mask_* field goes None"
    )
    assert hasattr(explorer, "hud_mask_diagnostics"), (
        "hud_mask_diagnostics() must be reachable from the scored policy's explorer"
    )
    for grid in grids:
        explorer._ingest(_Frame(np.asarray(grid)))
    return explorer, explorer.hud_mask_diagnostics()


def _counter_grids(n: int):
    """r11l's shape: a vertical monotone counter in column 0 that fills one cell per action. This is
    the frame family on which the REPAIRED edge-bar detector resolves a mask and the shipped
    classifier resolves nothing -- i.e. the lever's strongest firing."""

    import numpy as np

    out = []
    for filled in range(n):
        grid = np.full((64, 64), 3, dtype=int)
        grid[:, 0] = 0
        if filled:
            grid[:filled, 0] = 5
        grid[30:34, 30:34] = 7  # an interior board object that must never be masked
        out.append(grid)
    return out


def _shipped_bar_grids(n: int):
    """A horizontal top-row status bar: the ALREADY-SHIPPED `status_bar_classifier` resolves this
    one, and the repair adds no cells. The lever must read NOT fired here even though a mask IS
    resolved -- so a stub that returns a constant cannot satisfy both this and `_counter_grids`."""

    import numpy as np

    out = []
    for filled in range(n):
        grid = np.full((64, 64), 3, dtype=int)
        grid[0, :] = 8
        if filled:
            grid[0, :filled] = 5
        grid[30:34, 30:34] = 7
        out.append(grid)
    return out


@pytest.mark.memory_watchdog_skip
def test_hud_row_fields_are_populated_by_a_live_scored_explorer_not_by_the_test():
    """THE MEASURED GAP THIS CLOSES. `hud_mask_resolved` / `hud_mask_cell_count` /
    `hud_mask_source` read `None` -- not `False` -- in all 805 recorded rows of
    `results/outer_loop_scored_path_lever_ab_llm_on_20260726.json`, because the harness wrote them
    ONLY nested under `row["lever2_hud_fire"]` while every consumer written against
    `experiment_5836_frontier_discipline_ab.run_cell`'s schema reads them FLAT at row top level.
    `None` was the tell: nothing was unmeasured on the scored path, it was unreadable at the address
    the readers use.

    WHY THIS TEST CANNOT PASS AGAINST A DEAD CHANNEL, unlike its predecessor:

    1. Every asserted value is produced by a live `hud_mask_diagnostics()` call reached through
       `E3AgentPolicy.explorer`. The test supplies FRAMES; it never writes a hud_mask_* key.
    2. The INGEST WITNESS (`unique_frames`, `graph_nodes`) is asserted non-zero. A channel that was
       never fed -- the exact shape of the 0-of-122-nodes observe-channel defect -- reports zero
       here and fails.
    3. The reported cell count and digest are cross-checked against the explorer's OWN
       `hud_mask` array by an independent code path (`mask_summary`). A diagnostics stub, or a row
       whose fields were hand-set, cannot satisfy that.
    4. Two live configurations that must give OPPOSITE fire answers are asserted, so a constant
       (always-True or always-False) implementation fails one of them.

    All four mutations were RUN and confirmed to fail this test (2026-07-26): flat keys dropped from
    the projection, the projection unwired from run_cell, hud_mask_diagnostics replaced by a
    plausible constant stub, and _ingest made a no-op.
    """

    from carnot.agentic.arc_competition_agent import mask_summary

    m = _harness()

    # ---- the FIRING configuration: mask appears where the shipped classifier resolves none ----
    explorer, diag = _live_scored_explorer(_counter_grids(24))

    # (2) INGEST WITNESS. Zero here means nothing was ever fed to the detector, so any downstream
    # zero is a wiring defect rather than a null result.
    assert diag["unique_frames"] == 24, "the detector must actually have seen the frames"
    assert diag["graph_nodes"] >= 1

    # (3) The diagnostics must describe the explorer's REAL mask, cross-checked independently.
    assert explorer.hud_mask is not None, "a mask must actually exist on the live explorer"
    assert diag["hud_mask_cell_count"] == int(explorer.hud_mask.sum())
    assert diag["hud_mask_digest"] == mask_summary(explorer.hud_mask)["digest"]

    fields = m.hud_row_fields(diag)

    # (1) THE REGRESSION ITSELF: the three fields that were None must be present and populated.
    for key in ("hud_mask_resolved", "hud_mask_cell_count", "hud_mask_source"):
        assert key in fields, f"{key} must be emitted FLAT on the row, not only nested"
        assert fields[key] is not None, (
            f"{key} is None -- that is the unreadable-vs-resolved-nothing ambiguity this fix "
            "exists to remove, and it is not distinguishable from a real negative"
        )
    assert fields["hud_mask_resolved"] is True
    assert fields["hud_mask_cell_count"] > 0
    assert fields["hud_mask_source"].startswith("edge_bar_detector_req5960")
    assert fields["hud_diagnostics_readable"] is True
    assert fields["hud_diagnostics_error"] is None
    assert fields["lever2_fired"] is True, (
        "a mask APPEARING where the shipped config resolved none is the lever's strongest firing"
    )
    assert fields["lever2_fired_predicate"] == m.LEVER2_FIRE_PREDICATE_VERSION

    # Every flat key exp5836's rows carry must be emitted, so the two harnesses' rows are
    # interchangeable rather than accidentally similar.
    assert set(m.HUD_FLAT_ROW_KEYS) <= set(fields)

    # (4) THE OPPOSITE-ANSWER CONTROL. A resolved mask that the shipped classifier already produced
    # is NOT a firing. Same live machinery, different frames, and the answer must flip.
    _, shipped_diag = _live_scored_explorer(_shipped_bar_grids(24))
    shipped_fields = m.hud_row_fields(shipped_diag)
    assert shipped_fields["hud_mask_resolved"] is True, "a mask IS resolved in this arm"
    assert shipped_fields["hud_diagnostics_readable"] is True
    assert shipped_fields["hud_mask_digest"] == shipped_fields["hud_shipped_mask_digest"]
    assert shipped_fields["lever2_fired"] is False, (
        "the repair added nothing here; counting this as a firing would collapse the predicate "
        "into 'resolved == fired'"
    )


@pytest.mark.memory_watchdog_skip
def test_run_cell_actually_calls_the_hud_projection():
    """A correct-but-unwired projection is still a dead channel. The population fix is only real if
    `run_cell` lifts the fields onto the row, so assert the wiring at source level -- the same guard
    the lever-3 verdict tokens get -- alongside the value-level test above."""

    m = _harness()
    source = open(m.__file__, encoding="utf-8").read()
    assert "row.update(hud_row_fields(hud))" in source, (
        "run_cell must lift the HUD diagnostics onto the row's TOP LEVEL; emitting them only "
        "nested under lever2_hud_fire is the defect that made 805 recorded rows read None"
    )
    # The crash path must be instrumented too: a row that omits the keys entirely reads as None,
    # and a crashed arm reading as a clean null across a whole condition is a shipped defect here.
    assert "row.update(hud_row_fields(crash_hud))" in source, (
        "the run_game exception path must emit the HUD fields too"
    )


@pytest.mark.memory_watchdog_skip
def test_the_analyser_scores_both_row_schemas_from_live_diagnostics():
    """THE MIRROR-IMAGE HALF OF THE SAME DEFECT, also measured. The analyser's
    `recomputed_lever2_fired` read `row["lever2_hud_fire"]` EXCLUSIVELY, so it returned False on all
    1713 flat-schema rows of `results/cptb_20260726_cells/*.jsonl.gz` -- a 100% silent zero on real
    recorded data. Both directions of the schema split are asserted here, and the inputs are again
    live diagnostics rather than literals.

    ALSO ASSERTED: an UNRECORDED `hud_shipped_mask_digest` must not be read as None. `digest != None`
    is true for every resolved mask, so that fallback would arithmetically force "fired" on every
    resolved cell -- 1058 of those 1713 rows, none of which recorded a shipped-side comparison at
    all. Missing evidence is not negative evidence in either direction.
    """

    import importlib.util
    from pathlib import Path

    m = _harness()
    analyser_path = (
        Path(__file__).resolve().parents[2] / "scripts" / "analyze_scored_path_lever_ab.py"
    )
    spec = importlib.util.spec_from_file_location("analyze_scored_path_lever_ab", analyser_path)
    assert spec is not None and spec.loader is not None
    A = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(A)

    _, diag = _live_scored_explorer(_counter_grids(24))
    flat = m.hud_row_fields(diag)
    assert flat["lever2_fired"] is True

    # (a) A row in the CURRENT (both-address) schema.
    both = dict(flat, lever2_hud_fire=dict(diag), game="synthetic", seed=1)
    assert A.backfill_hud_flat_fields(both) == "both"
    assert A.recomputed_lever2_fired(both) is True

    # (b) A row in the PRE-FIX nested-only schema -- what the 805 recorded rows look like. The flat
    # read is None BEFORE the back-fill; that is the bug, asserted rather than described.
    nested_only = {"lever2_hud_fire": dict(diag), "game": "synthetic", "seed": 1}
    assert nested_only.get("hud_mask_resolved") is None
    assert A.backfill_hud_flat_fields(nested_only) == "nested_only"
    assert nested_only["hud_mask_resolved"] is True, "back-fill must make the flat address readable"
    assert nested_only["hud_mask_cell_count"] == diag["hud_mask_cell_count"]
    assert nested_only["hud_flat_fields_backfilled_from_nested"] is True
    assert A.recomputed_lever2_fired(nested_only) is True

    # (c) A row in the exp5836/cptb FLAT-only schema. It must be SCORED, not silently read False.
    flat_only = {k: flat[k] for k in flat if k.startswith(("hud_", "unique"))}
    flat_only.update(game="synthetic", seed=1)
    assert "lever2_hud_fire" not in flat_only
    assert A.backfill_hud_flat_fields(flat_only) == "flat_only"
    assert A.recomputed_lever2_fired(flat_only) is True

    # (d) The cptb shape exactly: flat, and with NO shipped-side digest recorded at all. Unknown is
    # not None -- this must NOT read as fired.
    no_shipped = {k: v for k, v in flat_only.items() if k != "hud_shipped_mask_digest"}
    assert no_shipped["hud_mask_resolved"] is True and no_shipped["hud_mask_digest"]
    assert A.recomputed_lever2_fired(no_shipped) is False, (
        "an unrecorded shipped digest must not be read as None; that would force 'fired' on every "
        "resolved cell without a single shipped-side comparison happening"
    )
    assert m.hud_lever_fired(no_shipped) is False, "harness and analyser must agree on this rule"

    # (e) A row with NO hud diagnostics anywhere carries no evidence in either direction.
    assert A.backfill_hud_flat_fields({"game": "synthetic", "seed": 1}) == "absent"

    # The two duplicated predicates must agree on live data, or the artifact's
    # recomputed-vs-stamped disagreement column becomes uninterpretable.
    for row in (both, nested_only, flat_only):
        assert A.recomputed_lever2_fired(row) == m.hud_lever_fired(row)


# --- REQ-ARC-WMTE-6640 / SCENARIO-ARC-WMTE-6640-5 (supervisor ledger plumbing) ---


def test_scenario_6640_5_supervisor_row_field_never_absent_never_none():
    """SCENARIO-ARC-WMTE-6640-5: the helper states the off case, converts a
    raising diagnostics call to an error marker, and refuses a non-dict. A
    None or missing value would read as zero to a flat consumer -- the exact
    ambiguity that made the 2026-08-21 supervisor A/B unreadable."""

    m = _harness()

    class _Off:
        def trajectory_supervisor_diagnostics(self):
            return {"enabled": False}

    class _Raises:
        def trajectory_supervisor_diagnostics(self):
            raise RuntimeError("boom")

    class _NonDict:
        def trajectory_supervisor_diagnostics(self):
            return None

    assert m.supervisor_row_field(_Off()) == {"enabled": False}
    assert m.supervisor_row_field(_Raises()) == {"error": "RuntimeError:boom"}
    assert m.supervisor_row_field(_NonDict()) == {"error": "non_dict_diagnostics:NoneType"}


def test_scenario_6640_5_run_cell_emits_the_field_on_both_paths():
    """SCENARIO-ARC-WMTE-6640-5: `run_cell` assigns `trajectory_supervisor`
    on the crash path AND the success path, both through the one shared
    helper. Source-level pin, same style as the induce-note wiring guard:
    the crash path returns before `row["wall_s"]` is ever assigned, so the
    two sides of that split are the two paths."""

    source = _HARNESS.read_text()
    call = 'row["trajectory_supervisor"] = supervisor_row_field(policy)'
    head, tail = source.split('row["wall_s"]', 1)
    assert call in head, "the CRASH path must carry the supervisor ledger too"
    assert call in tail, "the success path must carry the supervisor ledger"
    assert source.count(call) == 2, "exactly one assignment per path, via the shared helper"

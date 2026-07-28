"""REQ-ARC-WMTE-6017: the swallow guard's unmeasurable verdict, and its corpus scope.

WHAT WENT WRONG (two separate defects, both located by reproducing the LIVE path)
--------------------------------------------------------------------------------
1. UNMEASURABLE READ AS CLEAN. `hud_mask_swallow_check`'s docstring says it outright:
   "`swallows=False` with `n_changed_cells_total == 0` is reported as `no_dynamics_to_swallow`
   -- an unmeasurable verdict, NOT a clean one, so a consumer cannot read 'we checked and it
   is fine' off a corpus where the check could not fire." Every consumer nonetheless tested
   `rec.get("swallows")` for truthiness, so the unmeasurable verdict fell through to
   `hud_mask_status = "requested"` and the mask WAS applied. The function documented the trap
   and its only callers walked into it.

   REAL, ON-DISK INSTANCE -- not a constructed one: in
   `results/experiment_6011_world_model_change_gate_four_arm.json`, ft09 at all three seeds
   recorded `hud_mask_swallow.reason == "no_dynamics_to_swallow"` together with
   `hud_mask_status == "applied"`, a 64-cell mask, and `legacy_accuracy == 1.0` for the
   IDENTITY engine -- passing at the documented 0.5 threshold AND at the live 1.0 one.

2. THE VERDICT IS CORPUS-SCOPED, and the threshold was calibrated on a corpus shape the live
   path never judges. Same game, same 64-cell mask, two real corpora:

     lf52, 120 random actions (`collect_transitions`) : overlap 1.0000, 120/120 changing
        transitions deleted -> REFUSED (`no_changed_cells_outside_mask_cannot_distinguish`)
     lf52, live episode (25 rows, the shipped scored path's first induction attempt) :
        overlap 0.3086, 56 of 81 changed cells OUTSIDE the mask -> `ok`, applied

   Both records are honest, and the live one is POSITIVE EVIDENCE that the mask does not cover
   lf52's game state. So "the mask swallows lf52" is not a property of the mask; it is a
   property of (mask, corpus), and the 120-random-action corpus simply never moved anything but
   the counter.

WHAT IS PINNED HERE
-------------------
Every test runs against REAL transitions and a REAL mask, frozen under
`tests/fixtures/arc_hud_mask_swallow/` (see MANIFEST.json for the exact capture command and a
sha256 per file). Frozen rather than collected at test time because `environment_files/` is
untracked: a test that called `collect_transitions` would FAIL on a clean checkout instead of
asserting, and CLAUDE.md forbids the skip that would otherwise paper over that.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from carnot.agentic import arc_executable_world_model as e3
from carnot.agentic import arc_world_model_trust_energy as te

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "arc_hud_mask_swallow"


def _load(name: str):
    """Real (transitions, logical_mask) as captured from the real game/live path."""

    z = np.load(FIXTURES / f"{name}.npz")
    mask_raw = z["mask"]
    mask = None if mask_raw.size == 0 else mask_raw.astype(bool)
    rows = [
        e3.Transition(
            grid=z["grid"][i],
            action=int(z["action"][i]),
            next_grid=z["next_grid"][i],
            data=None,
            level_before=0,
            level_after=0,
        )
        for i in range(len(z["action"]))
    ]
    return rows, mask


def _identity(grid, action, data):
    return grid


# --------------------------------------------------------------------- DEFECT 1: unmeasurable
def test_unmeasurable_verdict_is_refused_not_read_as_clean_ft09_real_corpus() -> None:
    """MUST-FIRE on the fix's OWN origin incident, on ft09's real transitions.

    ft09 resolves a real 64-cell mask over a real corpus with ZERO state-changing transitions.
    The guard therefore cannot fire, which is `no_dynamics_to_swallow` -- and before this fix
    that unmeasurable verdict was applied as if it had been measured and cleared.
    """

    rows, mask = _load("ft09_offline120")
    assert mask is not None and int(mask.sum()) == 64, "fixture precondition: a real mask"

    rec = e3.hud_mask_swallow_check(rows, mask)
    assert rec["checked"] is True, rec
    assert rec["reason"] == "no_dynamics_to_swallow", rec
    assert rec["raw_changing_transitions"] == 0, "the corpus is why it is unmeasurable"
    assert rec["n_transitions"] == 120, "and it is unmeasurable over 120 transitions, not 0"

    # MUTATION PROOF, in-line: this is EXACTLY the value the pre-fix consumer tested. It is
    # falsy, so the pre-fix `elif self.hud_mask_swallow.get("swallows"):` did not fire and the
    # mask was applied -- reproducing exp6011's three ft09 rows.
    assert bool(rec.get("swallows")) is False, (
        "the pre-fix predicate reads this record as clean; that is the defect, and it is why "
        "truthiness on `swallows` is not a sufficient consumer test"
    )
    # The fix's predicate demands the AFFIRMATIVE reason.
    assert e3.hud_mask_swallow_clean(rec) is False, rec

    v = e3.WorldModelVerifier(rows, hud_mask=mask, hud_mask_enabled=True)
    assert v.hud_mask_status == "refused_swallow_check_unmeasurable", v.hud_mask_status
    assert v.hud_mask is None, "an unmeasurable mask must not reach the comparison"

    gate = e3.change_gate_decision(v.score(_identity), enabled=True)
    assert gate["hud_mask_swallow_refused_unmeasurable"] is True, gate
    assert gate["hud_mask_swallow_guard_fired"] is False, (
        "the two refusals are DIFFERENT claims: this one says the corpus could not measure "
        "the mask, not that the mask was measured to cover the dynamics"
    )
    assert gate["hud_mask_status"] == "refused_swallow_check_unmeasurable", gate


def test_unmeasurable_mask_launders_a_hud_scribbling_engine_on_ft09_real_transitions() -> None:
    """WHY refusing matters, on real transitions: the mechanism the fix closes.

    HONEST SCOPE, stated before the demonstration: on ft09's real corpus the IDENTITY engine
    scores 1.0 with the mask ON and 1.0 with it OFF (the corpus is all no-ops), so the fix
    flips NO admission there -- exp6011's ft09 rows are a record defect, not a measured
    laundering. The laundering is reachable with an engine that scribbles inside the masked
    cells, which is what this test builds: a real corpus, a real mask, and an engine whose
    only errors are hidden by the mask.
    """

    rows, mask = _load("ft09_offline120")

    def _hud_scribbler(grid, action, data):
        out = np.asarray(grid).copy()
        out[mask] = 7  # wrong everywhere the mask will erase, correct nowhere else
        return out

    # The pre-fix behaviour, reproduced through the public pre-computation hook: a caller-
    # supplied record that says "clean" is exactly what the old truthiness test inferred.
    as_if_clean = e3.WorldModelVerifier(
        rows,
        hud_mask=mask,
        hud_mask_enabled=True,
        hud_mask_swallow={"checked": True, "swallows": False, "reason": "ok"},
    ).score(_hud_scribbler)
    assert as_if_clean.hud_mask_status == "applied"
    assert as_if_clean.accuracy == 1.0, (
        "with the mask applied the scribbler is graded as perfect: every cell it got wrong is "
        "a cell the mask erases from both sides of the comparison"
    )

    # The fixed path refuses the unmeasurable mask, and the same engine is graded honestly.
    fixed = e3.WorldModelVerifier(rows, hud_mask=mask, hud_mask_enabled=True).score(_hud_scribbler)
    assert fixed.hud_mask_status == "refused_swallow_check_unmeasurable"
    assert fixed.accuracy == 0.0, (
        "unmasked, the scribbler is wrong about every transition -- 1.0 -> 0.0 is the "
        f"admission the unmeasurable-as-clean read was handing out. got={fixed.accuracy}"
    )


def test_select_trusted_world_model_also_refuses_an_unmeasurable_mask() -> None:
    """The second consumer had the same truthiness bug, and it feeds EVERY comparator.

    `select_trusted_world_model` pre-computes the verdict on the whole corpus and hands it to
    the held-out verifiers, so a wrong verdict there is wrong for the accuracy scores, both
    change-weighted consistencies, the off-path verifier and the change gate at once.
    """

    rows, mask = _load("ft09_offline120")

    def _hud_scribbler(grid, action, data):
        out = np.asarray(grid).copy()
        out[mask] = 7
        return out

    sel = te.select_trusted_world_model(
        rows,
        [te.WorldModelCandidate(name="scribbler", engine=_hud_scribbler, is_level_complete=None)],
        hidden_state=True,
        hud_mask=mask,
        hud_mask_enabled=True,
    )
    status = sel.selected_score.change_gate.get("hud_mask_status")
    assert status == "refused_swallow_check_unmeasurable", (
        "an unmeasurable mask must not be applied; before the fix this read 'applied', and "
        f"before the second half of the fix a refusal read 'disabled'. got={status}"
    )
    assert sel.selected_score.change_gate.get("hud_mask_swallow_source") == (
        "precomputed_by_caller"
    ), "the held-out verifiers must be told the WHOLE-corpus verdict, not re-derive one"

    # THE COMPARATORS, not just the record. `_score_accuracy` builds its own verifier on its
    # own slice and cannot be handed the verdict, so the refusal has to reach it through the
    # nulled mask. This engine is wrong only where the mask would erase it: masked it grades
    # 1.0, unmasked 0.0. Asserting the STATUS alone does not test this -- the status is
    # produced by the record-bearing verifier's own guard and stays correct even if the
    # comparators are still masked.
    assert sel.selected_score.heldout_accuracy == 0.0, (
        "the comparator still graded through the refused mask: an engine wrong about every "
        f"transition scored {sel.selected_score.heldout_accuracy}"
    )
    assert sel.hud_mask_enabled is False, (
        "no masking happened, so the selection must not claim the mask was enabled"
    )


# ------------------------------------------------------- MUST-FIRE on the calibration corpus
@pytest.mark.parametrize(
    "fixture,expected_reason",
    [
        ("lf52_offline120", "no_changed_cells_outside_mask_cannot_distinguish"),
        ("su15_offline120", "mask_overlaps_majority_of_changed_cells"),
        ("tn36_live_episode", "no_changed_cells_outside_mask_cannot_distinguish"),
    ],
)
def test_guard_refuses_the_masks_it_was_built_to_refuse_on_real_transitions(
    fixture: str, expected_reason: str
) -> None:
    """MUST-FIRE, on the real corpora the guard's own origin incident was measured on.

    lf52 and su15 are the two games named in the origin comment; tn36 is the ONE refusal the
    2026-07-27 live four-arm run actually recorded (`refused_swallows_dynamics` in both mask
    arms), so it pins the live path's own refusal too.
    """

    rows, mask = _load(fixture)
    rec = e3.hud_mask_swallow_check(rows, mask)
    assert rec["swallows"] is True, rec
    assert rec["reason"] == expected_reason, rec

    v = e3.WorldModelVerifier(rows, hud_mask=mask, hud_mask_enabled=True)
    assert v.hud_mask_status == "refused_swallows_dynamics", v.hud_mask_status
    assert v.hud_mask is None
    gate = e3.change_gate_decision(v.score(_identity), enabled=True)
    assert gate["hud_mask_swallow_guard_fired"] is True, gate
    assert gate["hud_mask_swallow_refused_unmeasurable"] is False, (
        "a MEASURED swallow must not be reported as an unmeasurable one"
    )


# ----------------------------------------------------------------------------- MUST-NOT-FIRE
def test_guard_does_not_refuse_the_noop_heavy_honest_mask_lp85_real_transitions() -> None:
    """MUST-NOT-FIRE on the exact false-positive case the source comment names.

    lp85 is the no-op-heavy corpus the pre-computation note calls out (~87 no-ops to 33
    changing). Its real mask erases 216 of 10917 changed cells and deletes NOT ONE of the 33
    changing transitions. A guard that refused this would disable the repair on precisely the
    corpus shape the repair exists for.
    """

    rows, mask = _load("lp85_offline120")
    rec = e3.hud_mask_swallow_check(rows, mask)
    assert rec["reason"] == "ok", rec
    assert rec["swallows"] is False, rec
    assert rec["raw_changing_transitions"] == 33, rec
    assert rec["changing_transitions_deleted"] == 0, (
        "the honest witness: the mask removes no mechanic observation at all"
    )
    assert e3.hud_mask_swallow_clean(rec) is True

    v = e3.WorldModelVerifier(rows, hud_mask=mask, hud_mask_enabled=True)
    assert v.hud_mask_status == "requested"
    assert v.score(_identity).hud_mask_status == "applied"


def test_guard_does_not_refuse_lf52s_live_corpus_where_the_dynamics_survive_masking() -> None:
    """MUST-NOT-FIRE, and it is the test that REFUTES the premise this repair started from.

    The premise was: "lf52 is EXACTLY the game the offline lane measured at 1.0000 changed-cell
    overlap -- masking deletes 60 of 60 changing transitions, the corpus becomes DYNAMICS-FREE,
    and the IDENTITY ENGINE IS OPTIMAL", so lf52's live `applied` had to be a guard failure.

    On lf52's REAL LIVE corpus that is not what the mask does: 56 of 81 changed cells fall
    OUTSIDE the mask and 2 changing transitions survive it. The guard measured that and cleared
    the mask correctly. Forcing a refusal here to match the premise would be a gate fitted to a
    conclusion the data contradicts.
    """

    rows, mask = _load("lf52_live_episode")
    rec = e3.hud_mask_swallow_check(rows, mask)

    assert rec["reason"] == "ok", rec
    assert rec["swallows"] is False, rec
    assert rec["n_transitions"] == 25, "this is the live episode corpus, not the 120-row one"
    assert rec["n_changed_cells_total"] == 81 and rec["n_changed_cells_inside_mask"] == 25, rec
    assert rec["n_changed_cells_total"] - rec["n_changed_cells_inside_mask"] == 56, (
        "56 changed cells outside the mask is POSITIVE evidence the mask does not cover "
        "lf52's game state"
    )
    assert rec["masked_changing_transitions"] == 2, rec

    v = e3.WorldModelVerifier(rows, hud_mask=mask, hud_mask_enabled=True)
    assert v.hud_mask_status == "requested"
    assert v.score(_identity).hud_mask_status == "applied", (
        "the live four-arm run recorded `applied` on this cell and it was CORRECT"
    )


# ------------------------------------------------------------------- DEFECT 2: corpus scope
def test_the_swallow_verdict_is_corpus_scoped_and_the_record_says_which_corpus() -> None:
    """lf52's verdict FLIPS between its two real corpora, and the record must show why.

    Without the scope fields a reader sees `refused` in one artifact and `applied` in another
    for the same game and the same mask, with nothing in either row to reconcile them -- which
    is exactly how a correct live clearance came to be read as a guard failure.
    """

    off_rows, off_mask = _load("lf52_offline120")
    live_rows, live_mask = _load("lf52_live_episode")
    assert int(off_mask.sum()) == int(live_mask.sum()) == 64, "the SAME 64-cell mask size"

    off = e3.hud_mask_swallow_check(off_rows, off_mask)
    live = e3.hud_mask_swallow_check(live_rows, live_mask)

    assert off["swallows"] is True and live["swallows"] is False, (off["reason"], live["reason"])
    assert off["n_transitions"] == 120 and live["n_transitions"] == 25, (
        "the scope fields are what makes the flip legible instead of contradictory"
    )
    assert off["changed_cell_overlap"] == 1.0
    assert live["changed_cell_overlap"] == pytest.approx(0.308642, abs=1e-6)


def test_no_threshold_on_the_per_transition_witnesses_is_defensible_so_none_is_gated() -> None:
    """Why the new per-transition fields are RECORDED and not GATED ON.

    A threshold on either witness that refused su15's offline corpus while sparing lf52's live
    one would have to sit inside a window a few hundredths wide, fitted between two specific
    game-corpora -- the project's own recorded anti-pattern (`invented_changed_cells` is left
    ungated for the same reason). This test asserts the windows are that narrow AND that the
    guard's actual behaviour ignores them, so adding such a gate later breaks a test instead of
    silently shipping.
    """

    su15_off = e3.hud_mask_swallow_check(*_load("su15_offline120"))
    lf52_live = e3.hud_mask_swallow_check(*_load("lf52_live_episode"))
    assert su15_off["swallows"] is True and lf52_live["swallows"] is False

    # Window on the per-changing-transition mean overlap: (0.9228, 0.9647]
    per_gap = (
        su15_off["mean_changed_cell_overlap_per_changing_transition"]
        - lf52_live["mean_changed_cell_overlap_per_changing_transition"]
    )
    assert 0.0 < per_gap < 0.05, (
        "the refused corpus and the cleared one are within 0.05 on this statistic: "
        f"{su15_off['mean_changed_cell_overlap_per_changing_transition']} vs "
        f"{lf52_live['mean_changed_cell_overlap_per_changing_transition']}"
    )
    # Window on changing-transition survival: (0.0392, 0.0800]
    surv_gap = lf52_live["changing_transition_survival"] - su15_off["changing_transition_survival"]
    assert 0.0 < surv_gap < 0.05, (
        "and within 0.05 on survival too: "
        f"{lf52_live['changing_transition_survival']} vs "
        f"{su15_off['changing_transition_survival']}"
    )

    # The behavioural half: 92% of lf52's live changing transitions are deleted by the mask and
    # the guard STILL clears it. A survival gate anywhere at or above 0.08 breaks this.
    assert lf52_live["changing_transitions_deleted"] == 23
    assert lf52_live["changing_transition_survival"] == pytest.approx(0.08, abs=1e-6)
    live_rows, live_mask = _load("lf52_live_episode")
    v = e3.WorldModelVerifier(live_rows, hud_mask=live_mask, hud_mask_enabled=True)
    assert v.hud_mask_status == "requested", (
        "survival is recorded, not gated: the cell-level evidence says this mask is honest"
    )


# -------------------------------------------------------------------- no dead channels
def test_every_new_swallow_record_field_can_be_non_trivially_populated() -> None:
    """No dead channels. This project's census found 877 stat blocks with an `errors` key and
    ZERO non-zero values; a field that can only ever hold its default is that same failure.
    Each new field is asserted against a distinctive REAL value here."""

    rows, mask = _load("su15_offline120")

    # n_transitions is populated even when there is no mask to judge -- "handed 120 rows and
    # no mask" must not be recorded the same way as "handed nothing".
    no_mask = e3.hud_mask_swallow_check(rows, None)
    assert no_mask["reason"] == "no_mask" and no_mask["n_transitions"] == 120, no_mask
    empty = e3.hud_mask_swallow_check([], mask)
    assert empty["reason"] == "no_transitions" and empty["n_transitions"] == 0, empty

    # shape accounting: a real corpus judged against a wrong-shaped mask skips every row, and
    # says so, instead of reporting a clean-looking zero.
    wrong = np.zeros((3, 3), dtype=bool)
    wrong[0, :] = True
    mismatch = e3.hud_mask_swallow_check(rows, wrong)
    assert mismatch["n_transitions_skipped_shape_mismatch"] == 120, mismatch
    assert mismatch["n_transitions_shape_matched"] == 0, mismatch
    # ... and the pre-existing, more specific status is PRESERVED rather than collapsed into
    # the new unmeasurable one (test_arc_world_model_change_gate.py already asserts it).
    assert (
        e3.WorldModelVerifier(rows, hud_mask=wrong, hud_mask_enabled=True).hud_mask_status
        == "shape_mismatch"
    )

    full = e3.hud_mask_swallow_check(rows, mask)
    assert full["n_transitions_shape_matched"] == 120, full
    assert full["n_transitions_skipped_shape_mismatch"] == 0, full
    assert full["changing_transitions_deleted"] == 49, full
    assert full["changing_transition_survival"] == pytest.approx(0.039216, abs=1e-6), full

    # The aggregation-level point, on real data: pooling CELLS and averaging per TRANSITION
    # are different numbers, so the second cannot be inferred from the first.
    assert full["changed_cell_overlap"] == pytest.approx(0.73913, abs=1e-6)
    assert full["mean_changed_cell_overlap_per_changing_transition"] == pytest.approx(
        0.964706, abs=1e-6
    )
    assert full["mean_changed_cell_overlap_per_changing_transition"] != full["changed_cell_overlap"]

    # survival takes its middle values too, not only 0.0 and 1.0.
    assert e3.hud_mask_swallow_check(*_load("lf52_live_episode"))[
        "changing_transition_survival"
    ] == pytest.approx(0.08, abs=1e-6)
    assert (
        e3.hud_mask_swallow_check(*_load("lp85_offline120"))["changing_transition_survival"] == 1.0
    )


def test_swallow_verdict_source_distinguishes_precomputed_from_self_computed() -> None:
    """A pre-computed verdict is a claim about the CALLER's corpus; a self-computed one is a
    claim about this verifier's slice. Both are legitimate and they are different, so a
    reviewer reading a refusal off an artifact must be able to tell which they have."""

    rows, mask = _load("lp85_offline120")
    own = e3.WorldModelVerifier(rows, hud_mask=mask, hud_mask_enabled=True)
    assert own.hud_mask_swallow_source == "computed_on_this_corpus"
    assert own.score(_identity).hud_mask_swallow_source == "computed_on_this_corpus"

    handed = e3.WorldModelVerifier(
        rows,
        hud_mask=mask,
        hud_mask_enabled=True,
        hud_mask_swallow=e3.hud_mask_swallow_check(rows, mask),
    )
    assert handed.hud_mask_swallow_source == "precomputed_by_caller"
    gate = e3.change_gate_decision(handed.score(_identity), enabled=True)
    assert gate["hud_mask_swallow_source"] == "precomputed_by_caller", gate


def test_hud_mask_swallow_clean_defaults_to_refusing_an_unrecognised_reason() -> None:
    """Refuse-on-doubt, including on a reason a FUTURE requirement adds.

    `apply_hud_mask`'s asymmetry decides the default: over-masking destroys correctness while
    under-masking only costs efficiency. So an unrecognised verdict must not be clean -- the
    opposite default is how `no_dynamics_to_swallow` became "applied" in the first place.
    """

    assert e3.hud_mask_swallow_clean({"checked": True, "swallows": False, "reason": "ok"}) is True
    for rec in (
        {"checked": True, "swallows": False, "reason": "some_future_reason"},
        {"checked": False, "swallows": False, "reason": "ok"},
        {"checked": True, "swallows": True, "reason": "ok"},
        {"checked": True, "swallows": False},
        {},
        None,
    ):
        assert e3.hud_mask_swallow_clean(rec) is False, rec


def test_fixture_manifest_records_the_capture_provenance_for_every_corpus() -> None:
    """The fixtures ARE the evidence, so their provenance has to be readable next to them: a
    frozen corpus with no record of how it was captured cannot be re-derived or challenged."""

    manifest = json.loads((FIXTURES / "MANIFEST.json").read_text())
    for name in (
        "lf52_offline120",
        "su15_offline120",
        "lp85_offline120",
        "ft09_offline120",
        "lf52_live_episode",
        "su15_live_episode",
        "tn36_live_episode",
    ):
        entry = manifest[name]
        assert entry["corpus"] and entry["mask"] and entry["sha256"], entry
        assert entry["n_transitions"] > 0, entry
        rows, mask = _load(name)
        assert len(rows) == entry["n_transitions"], name
        assert int(0 if mask is None else mask.sum()) == entry["mask_cells"], name
        # The verdict recorded at capture must still be the verdict the code produces, or the
        # fixture has silently stopped testing what it was captured to test.
        now = e3.hud_mask_swallow_check(rows, mask)
        was = entry["swallow_check_at_capture"]
        assert now["reason"] == was["reason"], (name, now["reason"], was["reason"])
        assert now["swallows"] == was["swallows"], name


# -------------------------------------------- REQ-ARC-WMTE-6019: the ORIGIN corpus scope, n=60
#
# WHY THIS BLOCK EXISTS. The MUST-FIRE cases above run on `*_offline120` fixtures -- 120-action
# RE-CAPTURES. But the calibration table those cases exist to defend (ops/verifier_gaps.md
# "GAP-WM-HUD-MASK-SWALLOW", and the copy in arc_executable_world_model.py's guard comment) is
# measured at `collect_transitions(n=60, seed=0)`, and its own numbers only reproduce at n=60:
#
#     n     lf52 overlap / raw -> surviving      su15 overlap / raw -> surviving
#     60      1.0000   60 -> 0                     0.7568   28 -> 1   <- reproduces the table
#     120     1.0000  120 -> 0                     0.7391   51 -> 2
#
# So "the guard fires on its own origin incident" was asserted at a corpus shape the origin
# incident never used. Worse, the 2026-07-28 corrigendum that CORRECTED the table's corpus
# scope itself misstated it as 120 -- a corrigendum whose subject is corpus scope getting the
# corpus wrong.
#
# No new fixture is needed, and that is a measured claim rather than a convenience: n=60 is a
# strict PREFIX of n=120 under the same seed (the same rng draws the same action sequence) and
# the mask is computed from the RESET FRAME ALONE, so it is byte-identical between the two.
# Verified 2026-07-28 by running a live `collect_transitions(n=60, seed=0)` re-collect against
# these fixture slices: identical overlap, raw and surviving counts on lf52, su15 and lp85,
# under BOTH `edge_bar_detector` settings.
_N60 = 60


@pytest.mark.parametrize(
    "fixture,expected_reason,expected_raw,expected_overlap",
    [
        ("lf52_offline120", "no_changed_cells_outside_mask_cannot_distinguish", 60, 1.0000),
        ("su15_offline120", "mask_overlaps_majority_of_changed_cells", 28, 0.7568),
    ],
)
def test_guard_fires_on_the_ORIGIN_n60_corpus_not_only_the_120_recapture(
    fixture: str, expected_reason: str, expected_raw: int, expected_overlap: float
) -> None:
    """MUST-FIRE at the corpus scope the calibration table was actually measured on.

    The asserted `expected_raw` / `expected_overlap` are the TABLE'S OWN published numbers, so
    this test fails if either the guard's arithmetic drifts or the table is edited to values
    the code does not produce. That is the point: it makes the documentation checkable rather
    than merely present.
    """

    rows, mask = _load(fixture)
    assert len(rows) == 120, "fixture precondition: the 120-action capture"
    assert mask is not None and int(mask.sum()) == 64, "fixture precondition: a real mask"

    rec = e3.hud_mask_swallow_check(rows[:_N60], mask)

    # The published table row, re-derived.
    assert rec["n_transitions"] == _N60, rec
    assert rec["raw_changing_transitions"] == expected_raw, (
        f"the table publishes {expected_raw} raw changing transitions for {fixture} at n=60; "
        f"got {rec['raw_changing_transitions']}. Either the guard drifted or the table is wrong."
    )
    assert round(float(rec["changed_cell_overlap"]), 4) == expected_overlap, rec

    # The VERDICT -- unchanged from n=120, which is what makes the scope error a provenance
    # correction rather than a decision correction.
    assert rec["swallows"] is True, rec
    assert rec["reason"] == expected_reason, rec
    assert e3.hud_mask_swallow_clean(rec) is False

    v = e3.WorldModelVerifier(rows[:_N60], hud_mask=mask, hud_mask_enabled=True)
    assert v.hud_mask_status == "refused_swallows_dynamics", v.hud_mask_status
    assert v.hud_mask is None, "a refused mask must not be applied to grading"


def test_guard_does_NOT_fire_on_the_honest_mask_at_the_ORIGIN_n60_corpus_either() -> None:
    """MUST-NOT-FIRE at n=60, on the same false-positive case the source comment names.

    A guard that refused lp85 would disable the repair on exactly the no-op-heavy corpus shape
    the repair exists for. Pinned at BOTH scopes so a future threshold change cannot buy an
    n=120 MUST-FIRE by breaking the n=60 MUST-NOT-FIRE (or the reverse) without a test failing.
    """

    rows, mask = _load("lp85_offline120")
    rec = e3.hud_mask_swallow_check(rows[:_N60], mask)

    assert rec["n_transitions"] == _N60, rec
    assert rec["reason"] == "ok", rec
    assert rec["swallows"] is False, rec
    assert e3.hud_mask_swallow_clean(rec) is True, (
        "an affirmative clean verdict, not merely a non-refusal"
    )
    # The honest witness, at this scope: every changing transition survives the mask.
    assert rec["raw_changing_transitions"] == 16, rec
    assert rec["changing_transitions_deleted"] == 0, (
        "the mask removes no mechanic observation at n=60 either"
    )

    v = e3.WorldModelVerifier(rows[:_N60], hud_mask=mask, hud_mask_enabled=True)
    assert v.score(_identity).hud_mask_status == "applied"


def test_the_n60_verdicts_agree_with_the_n120_verdicts_so_scope_is_provenance_not_decision() -> (
    None
):
    """The claim the corrigendum rests on, asserted rather than asserted-in-prose.

    "The corpus-scope error is a provenance correction, not a verdict correction" is only
    honest if the DECISION is in fact stable across the two scopes. Checked here for all three
    offline fixtures at once, including the direction that would matter most if it broke: the
    honest mask staying honest.
    """

    for fixture, expect_swallows in (
        ("lf52_offline120", True),
        ("su15_offline120", True),
        ("lp85_offline120", False),
        ("ft09_offline120", False),
    ):
        rows, mask = _load(fixture)
        r60 = e3.hud_mask_swallow_check(rows[:_N60], mask)
        r120 = e3.hud_mask_swallow_check(rows, mask)
        assert r60["swallows"] is expect_swallows, (fixture, r60)
        assert r120["swallows"] is expect_swallows, (fixture, r120)
        assert r60["reason"] == r120["reason"], (
            f"{fixture}: the REASON differs between n=60 and n=120 ({r60['reason']} vs "
            f"{r120['reason']}), so the scope error is NOT purely a provenance issue and the "
            "corrigendum's claim would need narrowing"
        )
        assert e3.hud_mask_swallow_clean(r60) == e3.hud_mask_swallow_clean(r120), (fixture,)
        # ft09 is the unmeasurable case at BOTH scopes -- pinned so a corpus slice cannot
        # accidentally turn "could not check" into "checked and clean".
        if fixture == "ft09_offline120":
            assert r60["reason"] == "no_dynamics_to_swallow", r60
            assert r60["raw_changing_transitions"] == 0, r60

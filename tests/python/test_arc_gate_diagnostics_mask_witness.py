"""REQ-ARC-WMTE-6014: the per-attempt gate diagnostics must carry the MASK/GATE witnesses.

WHY THIS TEST EXISTS
--------------------
`generator_liveness_witness()["induction_attempt_gate_diagnostics"]` is a read-only
projection of a fixed key tuple off each entry in `self.induction_attempts`. Before
REQ-6014 that tuple named the skip reason and the trust margin but NOT whether a HUD mask
had actually been applied, nor the symmetric union-fidelity quantity the REQ-6011/-6013
change gate actually decides on.

That gap is not cosmetic for the four-arm mask/gate matrix. A mask-arm cell whose mask
silently failed to resolve produced a row that was byte-indistinguishable from a cell that
masked correctly and made no difference -- so the arm's own null could not be attributed
to either "the treatment was applied and did nothing" or "the treatment never reached this
cell". This project has shipped that exact confusion before (the 2026-07-27 first-win
measurement's 174/174 `induction_attempts_planned == 0`, where every LLM-on arm came out
bit-identical to its control and p=1.0 was an arithmetic identity, not a measurement).

WHAT IS PINNED HERE
-------------------
1. Every one of the six new keys can be NON-TRIVIALLY populated. This project ran a census
   that found 877 stat blocks carrying an `errors` key with zero non-zero values across the
   entire corpus -- a structurally dead channel that looked like a clean signal. A new
   diagnostic field that can only ever be null/absent is that same failure, so each key is
   asserted against a distinctive value here rather than merely asserted present.
2. The projection is DATA-DRIVEN, not a constant. A projection hard-coded to emit the
   witness keys with fixed values would pass an "is the key there?" test while reporting
   the same thing for every cell; the two-attempt case below pins that the values track
   their source attempts independently.
3. The `if k in a` guard still holds: an attempt dict that never wrote a key must OMIT it,
   not emit None. Absent and null are different claims -- null reads as "measured, and it
   was nothing".

   HISTORICAL NOTE (REQ-ARC-WMTE-6019, 2026-07-28): item 3 used to name the hidden-state
   branch as the example of a branch that "does not write `hud_mask_status`". That WAS true
   and was itself a defect, not a design: measured over the 2026-07-27 four-arm run,
   `hud_mask_status` is absent from all 44 hidden-state attempts (11 games x 4 arms) and
   present on all 56 others, so REQ-6017's refusal-naming fix
   (`refused_swallows_dynamics` / `refused_swallow_check_unmeasurable`) could not be READ on
   any 0.08-wall game. The branch now writes it from `trust_score.change_gate`, and the
   absent-!=-null discipline is pinned below against a genuinely-bare attempt dict instead
   of against that defect.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Importing arc_competition_agent costs ~680MB of module state, which the memory watchdog
# reads as a per-test leak. This is the SAME exemption the sibling witness suite uses
# (test_arc_scored_path_liveness_witness.py:56) and it is NOT a test skip -- every test in
# this file runs and asserts. CLAUDE.md forbids skipped tests; it does not forbid exempting
# a heavy-import suite from a leak heuristic that is measuring the import, not a leak.
pytestmark = pytest.mark.memory_watchdog_skip

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "python") not in sys.path:
    sys.path.insert(0, str(REPO / "python"))


def _policy(game: str = "zz00"):
    from carnot.agentic.arc_competition_agent import E3AgentPolicy

    return E3AgentPolicy(game)


def _plain_branch_attempt() -> dict:
    """An attempt dict shaped like the one the NON-hidden-state branch writes.

    The keys and their spellings are copied from the branch itself
    (arc_competition_agent.py, the `else:` arm of `if self.short in HIDDEN_STATE_GAME_IDS`)
    rather than invented, so a rename in the source breaks this test instead of silently
    making the projection emit nothing.
    """

    return {
        "planned": False,
        "skipped": "world_model_change_gate_degenerate_engine_no_correct_changed_cells",
        "trust_metric": "exact",
        "verify_accuracy": 0.96,
        "verify_cell_recall": 0.0,
        "hud_mask_status": "applied",
        "hud_mask_cells": 64,
        "hud_mask_reason": "explorer_mask_latched",
        "verify_change_fidelity": 0.0,
        "verify_spurious_changed_cells": 17,
    }


def _hidden_state_attempt() -> dict:
    """An attempt dict shaped like the HIDDEN_STATE_GAME_IDS branch's.

    REQ-ARC-WMTE-6019: this now CARRIES `hud_mask_status`/`hud_mask_cells`, read from
    `trust_score.change_gate` by the branch itself. Before 6019 the branch wrote only
    `hud_mask_reason`, which made every named refusal on the 11 hidden-state games (every
    0.08-wall game) unreadable from a cell record. The status value used here is a REFUSAL
    name, not `applied`, because the refusal is the case that was invisible.
    """

    return {
        "planned": False,
        "skipped": "hidden_state_change_gate_degenerate_engine_no_correct_changed_cells",
        "trust_energy": 497.756429,
        "heldout_accuracy": 0.625,
        "heldout_change_consistency": 0.0,
        "correct_changed_cells": 0,
        "binary_gate_pass": True,
        "verify_change_fidelity": 0.0,
        "verify_spurious_changed_cells": 3,
        "change_gate_hidden_state_enabled": True,
        "hud_mask_reason": "resolved",
        "hud_mask_status": "refused_swallow_check_unmeasurable",
        "hud_mask_cells": 64,
    }


def test_every_new_witness_key_can_be_non_trivially_populated() -> None:
    """No dead channels: each of the six new keys must survive the projection with the
    distinctive value its source attempt carried."""
    policy = _policy()
    policy.induction_attempts = [_plain_branch_attempt()]

    diags = policy.generator_liveness_witness()["induction_attempt_gate_diagnostics"]
    assert len(diags) == 1, diags
    d = diags[0]

    # The four plain-branch witnesses, each against its own distinctive value.
    assert d["hud_mask_status"] == "applied", (
        "hud_mask_status is the ONLY proof the mask treatment reached this cell; without "
        f"it a silently-unresolved mask is indistinguishable from a no-op mask. got={d}"
    )
    assert d["hud_mask_cells"] == 64, d
    assert d["hud_mask_reason"] == "explorer_mask_latched", d
    assert d["verify_change_fidelity"] == 0.0, d
    assert d["verify_spurious_changed_cells"] == 17, (
        "the spurious-write count is the direction cell_recall structurally cannot see "
        f"(it masks to TRUE changes only), so it must survive the projection. got={d}"
    )

    # And the pre-existing keys are untouched -- this REQ is additive.
    assert d["verify_accuracy"] == 0.96, d
    assert d["trust_metric"] == "exact", d


def test_hidden_state_witness_carries_the_follow_flag_and_the_named_mask_status() -> None:
    """REQ-6013 resolves the hidden-state gate by FOLLOWING the -6011 flag. A default is not
    a guarantee, so the resolved value must be RECORDED -- these are the 11 games that carry
    every 0.08-wall cell, and a stale ambient override would turn a gate arm back into a
    control arm with no trace in the row."""
    policy = _policy()
    policy.induction_attempts = [_hidden_state_attempt()]

    d = policy.generator_liveness_witness()["induction_attempt_gate_diagnostics"][0]

    assert d["change_gate_hidden_state_enabled"] is True, (
        "the follow-the-6011-flag resolution must be observed in the row, not assumed "
        f"from the docstring. got={d}"
    )
    assert d["verify_change_fidelity"] == 0.0, d
    assert d["verify_spurious_changed_cells"] == 3, d
    assert d["correct_changed_cells"] == 0, d
    assert d["binary_gate_pass"] is True, (
        "the incumbent gate PASSING while correct_changed_cells==0 is the origin incident "
        f"itself; both must be visible side by side or the rejection cannot be checked. {d}"
    )

    # REQ-ARC-WMTE-6019: the refusal NAME must survive on this branch too. Before 6019 the
    # hidden-state branch wrote only `hud_mask_reason`, so on all 44 hidden-state attempts of
    # the 2026-07-27 run `hud_mask_status` was absent -- i.e. on every 0.08-wall game, a
    # measured refusal was indistinguishable from "no mask was ever requested".
    assert d["hud_mask_status"] == "refused_swallow_check_unmeasurable", (
        "the named refusal is the whole point of REQ-6017's fix, and it is DECIDED inside "
        "select_trusted_world_model -- if it does not reach the row, the fix is invisible "
        f"on exactly the games that matter most. got={d}"
    )
    assert d["hud_mask_cells"] == 64, (
        "a refusal with a real cell count is distinguishable from `disabled`/`unresolved` "
        f"(no mask at all); the count is what makes that distinction checkable. got={d}"
    )
    # ...and the reason IS written by both branches and must come through.
    assert d["hud_mask_reason"] == "resolved", d
    # The status and the reason are DIFFERENT claims and must not be collapsed: `resolved`
    # says the explorer produced a mask, `refused_*` says the guard then declined to use it.
    assert d["hud_mask_status"] != d["hud_mask_reason"], d


def test_projection_tracks_its_source_and_is_not_a_constant() -> None:
    """A projection hard-coded to emit the witness keys would pass a presence test while
    reporting the same thing for every cell. Two attempts with different values must come
    through as two different diagnostics, in order."""
    policy = _policy()
    a0 = _plain_branch_attempt()
    a1 = _plain_branch_attempt()
    a1["hud_mask_status"] = "unresolved"
    a1["hud_mask_cells"] = 0
    a1["verify_change_fidelity"] = 0.87
    a1["verify_spurious_changed_cells"] = 0
    policy.induction_attempts = [a0, a1]

    diags = policy.generator_liveness_witness()["induction_attempt_gate_diagnostics"]
    assert len(diags) == 2, diags
    assert diags[0]["hud_mask_status"] == "applied", diags
    assert diags[1]["hud_mask_status"] == "unresolved", diags
    assert diags[0]["hud_mask_cells"] == 64 and diags[1]["hud_mask_cells"] == 0, diags
    assert diags[0]["verify_change_fidelity"] != diags[1]["verify_change_fidelity"], (
        "both rows reported the same fidelity -- the projection is not reading its source"
    )


def test_witness_keys_are_the_ones_the_agent_actually_writes() -> None:
    """READ, DO NOT MODEL. The fixtures above are hand-written dicts, and two independent
    reconstructions of a wrong shape agreeing with each other is this project's canonical
    measurement failure. So assert the key SPELLINGS against the agent source itself: if a
    branch renames a field, the projection would silently emit nothing and every arm would
    report an absent witness that reads as 'the treatment was never applied'."""
    src = (REPO / "python" / "carnot" / "agentic" / "arc_competition_agent.py").read_text()
    for key in (
        "hud_mask_status",
        "hud_mask_cells",
        "hud_mask_reason",
        "verify_change_fidelity",
        "verify_spurious_changed_cells",
        "change_gate_hidden_state_enabled",
        "hud_mask_swallow",
    ):
        # Twice: once where a branch ASSIGNS it onto the attempt dict, once in the
        # projection tuple. One occurrence would mean the projection names a key nothing
        # writes (or vice versa).
        assert src.count(f'"{key}"') >= 2, (
            f"{key} appears < 2x in the agent source: the projection and the branch that "
            "writes it have diverged, so this witness is structurally dead"
        )


# ------------------------------------------------------------------ REQ-ARC-WMTE-6017
def test_the_swallow_record_survives_the_projection_it_was_computed_and_discarded() -> None:
    """REQ-ARC-WMTE-6017: `hud_mask_swallow` was a DEAD CHANNEL in the recorded rows.

    The guard computed a full auditable record on every attempt -- reason, the measured
    overlap, its threshold, the raw/masked changing-transition counts -- and the projection
    dropped all of it. Consequence, measured: in all 100 cells of
    `results/arc_wm_four_arm_20260727/cells/`, `hud_mask_swallow` is absent, so lf52's
    `hud_mask_status == "applied"` could be DATED from the record but never EXPLAINED from it,
    and an unmeasurable non-refusal was indistinguishable from a measured clearance. Both
    branches now lift the record onto the attempt and the projection carries it.
    """

    policy = _policy()
    plain = _plain_branch_attempt()
    # The real shape, copied from a real record (ft09, exp6011 seed 0) -- the unmeasurable
    # verdict that used to read as clean.
    plain["hud_mask_swallow"] = {
        "checked": True,
        "swallows": False,
        "reason": "no_dynamics_to_swallow",
        "changed_cell_overlap": 0.0,
        "overlap_threshold": 0.5,
        "raw_changing_transitions": 0,
        "masked_changing_transitions": 0,
        "n_transitions": 120,
    }
    hidden = _hidden_state_attempt()
    hidden["hud_mask_swallow"] = {
        "checked": True,
        "swallows": True,
        "reason": "mask_overlaps_majority_of_changed_cells",
        "changed_cell_overlap": 0.73913,
        "overlap_threshold": 0.5,
        "raw_changing_transitions": 51,
        "masked_changing_transitions": 2,
        "n_transitions": 120,
    }
    policy.induction_attempts = [plain, hidden]

    diags = policy.generator_liveness_witness()["induction_attempt_gate_diagnostics"]
    assert len(diags) == 2, diags

    # Non-trivially populated, and tracking its own source rather than a constant: the two
    # attempts carry DIFFERENT verdicts and both must come through intact.
    assert diags[0]["hud_mask_swallow"]["reason"] == "no_dynamics_to_swallow", diags[0]
    assert diags[1]["hud_mask_swallow"]["reason"] == "mask_overlaps_majority_of_changed_cells", (
        diags[1]
    )
    # The measurement, not just the verdict: a reviewer must be able to re-derive the decision.
    assert diags[1]["hud_mask_swallow"]["changed_cell_overlap"] == 0.73913, diags[1]
    assert diags[1]["hud_mask_swallow"]["overlap_threshold"] == 0.5, diags[1]
    assert diags[0]["hud_mask_swallow"]["raw_changing_transitions"] == 0, (
        "zero changing transitions is WHY the guard could not fire on that cell; without it "
        "the row cannot distinguish 'checked, clean' from 'could not check'"
    )
    assert diags[0]["hud_mask_swallow"]["n_transitions"] == 120, (
        "the corpus SCOPE: the same mask on the same game gets opposite verdicts on a "
        "120-row corpus and a 25-row one, so the verdict is unreadable without it"
    )

    # And the `if k in a` guard still holds: an attempt that never wrote the key omits it.
    bare = _policy()
    bare.induction_attempts = [_plain_branch_attempt()]
    assert (
        "hud_mask_swallow"
        not in bare.generator_liveness_witness()["induction_attempt_gate_diagnostics"][0]
    ), "absent != null; a null verdict would fabricate a check that never ran"


# ------------------------------------------------------------------ REQ-ARC-WMTE-6019
def test_absent_is_not_null_on_an_attempt_that_writes_no_mask_fields() -> None:
    """The absent-!=-null discipline, pinned against a GENUINELY bare attempt.

    This assertion used to be made against the hidden-state branch's output, on the premise
    that the branch "does not report a mask status". That premise was a DEFECT (REQ-6019: the
    status was decided and then dropped, invisible on all 44 hidden-state attempts of the
    2026-07-27 run), so pinning the discipline to it was pinning the bug in place. The
    discipline itself is still right and still needs a test, so it is asserted here against an
    attempt dict that legitimately wrote no mask fields at all -- e.g. an attempt that was
    skipped before any verifier was built.
    """

    policy = _policy()
    policy.induction_attempts = [{"planned": False, "skipped": "no_transitions_yet"}]

    d = policy.generator_liveness_witness()["induction_attempt_gate_diagnostics"][0]
    assert d["skipped"] == "no_transitions_yet", d
    for key in (
        "hud_mask_status",
        "hud_mask_cells",
        "hud_mask_reason",
        "hud_mask_swallow",
        "legacy_accuracy_would_pass_at_live_threshold",
        "noop_ok_is_vacuous",
    ):
        assert key not in d, (
            f"{key} came through as a null on an attempt that never measured it; a null "
            f"reads as 'we looked and it was nothing', which is a fabricated measurement. {d}"
        )


def test_in_arm_admission_counterfactual_and_noop_vacuity_survive_the_projection() -> None:
    """REQ-ARC-WMTE-6019: two more fields computed on every attempt and then discarded.

    `change_gate_decision` emits both on EVERY gate record, but `change_gate` itself is not in
    the projection tuple, so both were absent from all 104 attempts of the 2026-07-27 four-arm
    run. Consequences, each concrete:

      legacy_accuracy_would_pass_at_live_threshold -- the IN-ARM admission counterfactual at
        the threshold the agent actually ships (1.0). Without it, the four-arm artifact's
        admission reading had to be taken CROSS-ARM off the control row, which is a different
        engine (that run's per-arm LLM response counts are 93/91/94/89), i.e. a weaker claim
        than the data could have supported.
      noop_ok_is_vacuous -- whether the no-op channel could fire at all. `noop_ok` is
        `rate <= threshold` and the rate is 0.0 when `n_noop == 0`, so it passes VACUOUSLY on a
        no-op-free corpus and the gate can reach `passed=True` on an empty pass region. The
        disclosure is deliberately reported rather than gated (refusing would cost capability
        on a corpus property unrelated to engine quality -- the opposite asymmetry from the
        mask guard's), which only works if it reaches the row.

    Both are asserted with values that DIFFER between the two attempts, so a constant-emitting
    projection fails here rather than passing a presence check.
    """

    policy = _policy()
    plain = _plain_branch_attempt()
    plain["legacy_accuracy_would_pass_at_live_threshold"] = False
    plain["noop_ok_is_vacuous"] = True
    hidden = _hidden_state_attempt()
    hidden["legacy_accuracy_would_pass_at_live_threshold"] = True
    hidden["noop_ok_is_vacuous"] = False
    policy.induction_attempts = [plain, hidden]

    diags = policy.generator_liveness_witness()["induction_attempt_gate_diagnostics"]
    assert len(diags) == 2, diags

    assert diags[0]["legacy_accuracy_would_pass_at_live_threshold"] is False, diags[0]
    assert diags[1]["legacy_accuracy_would_pass_at_live_threshold"] is True, diags[1]
    assert diags[0]["noop_ok_is_vacuous"] is True, (
        "a vacuous no-op pass must be visible in the row, or a consumer reading `passed` "
        f"alone banks an empty pass region as evidence. got={diags[0]}"
    )
    assert diags[1]["noop_ok_is_vacuous"] is False, diags[1]
    # Data-driven, not a constant: the two rows disagree on both fields.
    assert (
        diags[0]["legacy_accuracy_would_pass_at_live_threshold"]
        != diags[1]["legacy_accuracy_would_pass_at_live_threshold"]
    ), diags
    assert diags[0]["noop_ok_is_vacuous"] != diags[1]["noop_ok_is_vacuous"], diags


def test_both_branches_actually_write_the_new_fields_read_the_source() -> None:
    """READ, DO NOT MODEL. The fixtures above are hand-written, and this project's canonical
    measurement failure is two reconstructions of a wrong shape agreeing with each other.

    So assert against the AGENT SOURCE that the real branches write what the projection reads.
    `legacy_accuracy_would_pass_at_live_threshold` / `noop_ok_is_vacuous` are lifted by a loop
    over a key tuple in BOTH branches, so each name must appear at least three times: the two
    branch lifts plus the projection tuple. `hud_mask_status` must be assigned in the
    hidden-state branch specifically -- the defect REQ-6019 fixed was that only the plain
    branch assigned it.
    """

    src = (REPO / "python" / "carnot" / "agentic" / "arc_competition_agent.py").read_text()

    for key in ("legacy_accuracy_would_pass_at_live_threshold", "noop_ok_is_vacuous"):
        assert src.count(f'"{key}"') >= 3, (
            f"{key} appears < 3x: expected two branch lifts + the projection tuple. A "
            "projection naming a key no branch writes is a structurally dead channel."
        )

    # The hidden-state branch reads the status out of its OWN gate dict. Pin the exact
    # expression so a rename of `_hs_change_gate` cannot silently re-open the hole.
    assert 'attempt["hud_mask_status"] = str(_hs_change_gate["hud_mask_status"])' in src, (
        "the hidden-state branch no longer lifts hud_mask_status from its change_gate dict; "
        "REQ-6019's hole (status absent on all 44 hidden-state attempts) is re-opened"
    )
    assert 'if "hud_mask_status" in _hs_change_gate:' in src, (
        "the lift must be presence-guarded so a gate dict without the key yields an ABSENT "
        "field rather than a fabricated null"
    )

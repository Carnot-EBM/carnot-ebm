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
3. The `if k in a` guard still holds: an attempt dict that never wrote a key (the
   hidden-state branch does not write `hud_mask_status`) must OMIT it, not emit None.
   Absent and null are different claims -- null reads as "measured, and it was nothing".
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

    Deliberately carries NO `hud_mask_status`/`hud_mask_cells`: that branch records
    `hud_mask_reason` but grades through `select_trusted_world_model`, which does not
    publish a per-verifier mask status. The asymmetry is the point of the absence test.
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
        "hud_mask_reason": "explorer_mask_unresolved",
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


def test_hidden_state_witness_carries_the_follow_flag_and_omits_mask_status() -> None:
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

    # The absence half: this branch never writes hud_mask_status, so the projection must
    # OMIT the key rather than emit None. `None` would read as "we looked and there was no
    # mask", which is a different and false claim from "this branch does not report one".
    assert "hud_mask_status" not in d, (
        f"absent != null; a null here would fabricate a measurement that never happened. {d}"
    )
    assert "hud_mask_cells" not in d, d
    # ...while the reason IS written by both branches and must come through.
    assert d["hud_mask_reason"] == "explorer_mask_unresolved", d


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
    ):
        # Twice: once where a branch ASSIGNS it onto the attempt dict, once in the
        # projection tuple. One occurrence would mean the projection names a key nothing
        # writes (or vice versa).
        assert src.count(f'"{key}"') >= 2, (
            f"{key} appears < 2x in the agent source: the projection and the branch that "
            "writes it have diverged, so this witness is structurally dead"
        )

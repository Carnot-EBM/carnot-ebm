"""Drift guards for the SHARED HUD row schema (``carnot.agentic.arc_hud_row_schema``).

THE DEFECT CLASS THESE TESTS EXIST TO MAKE IMPOSSIBLE (measured 2026-07-26, both directions).
The same quantity had two different addresses on a result row, and each reader saw a clean,
error-free ``None`` on the other writer's rows -- indistinguishable from "the detector resolved
nothing":

  * ``scripts/arc_scored_path_lever_harness.py`` wrote the HUD diagnostics NESTED, under
    ``row["lever2_hud_fire"]``. All 805 recorded rows of
    ``results/outer_loop_scored_path_lever_ab_llm_on_20260726.json`` read ``None`` at the FLAT
    address while the nested copy was populated in all 805 (True 598 / False 207).
  * ``python/carnot/experiment_5836_frontier_discipline_ab.py`` wrote them FLAT only. All 1713 rows
    of ``results/cptb_20260726_cells/*.jsonl.gz`` therefore scored ``lever2_fired = False`` under a
    nested-only reader.

Two independent projections of one source dict IS the defect, so the test that matters is not "does
each writer work" but "do the two writers produce the SAME key set, and does every consumer read a
row of either schema identically". The bug bit twice, in opposite directions, and neither writer's
own tests caught it -- because each tested itself against its own schema.

Also guarded here: the distinction between "the lever did not fire" (evidence) and "this row cannot
say" (no evidence). exp5836's hand-rolled projection OMITTED ``hud_shipped_mask_digest``, the
comparator the fire predicate needs. Back-fill cannot recover a field that was never written, so
those 1713 rows are permanently unscoreable and MUST be excluded from a lever-2 denominator rather
than counted as 1713 non-fires. ``lever2_scoreable`` is that distinction; a test that lets it
collapse would let a published rate be computed over a denominator that structurally could not
answer.

No network, no GPU, no game environment: pure projection + read logic.

Spec refs: REQ-ARC-WMTE-5960, SCENARIO-ARC-WMTE-5960-ROW-SCHEMA-PARITY.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "python") not in sys.path:
    sys.path.insert(0, str(REPO / "python"))

from carnot.agentic import arc_hud_row_schema as schema  # noqa: E402

_HARNESS = REPO / "scripts" / "arc_scored_path_lever_harness.py"
_ANALYZER = REPO / "scripts" / "analyze_scored_path_lever_ab.py"
_EXP5836 = REPO / "python" / "carnot" / "experiment_5836_frontier_discipline_ab.py"


def _load(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Copied verbatim from a recorded row of
# results/outer_loop_scored_path_lever_ab_llm_on_20260726.json (r11l, the game where the repaired
# detector resolves a 64-cell mask the shipped classifier does not resolve at all).
REAL_R11L_DIAGNOSTICS = {
    "hud_mask_resolved": True,
    "hud_mask_source": "edge_bar_detector_req5960_stage2_confirmed",
    "hud_mask_cell_count": 64,
    "hud_mask_digest": "fcbba0b6818499b6",
    "hud_shipped_mask_cell_count": 0,
    "hud_shipped_mask_digest": None,
    "unique_frames": 37,
    "graph_nodes": 41,
    "node_inflation_vs_unique_frames": 1.108,
    "collapse_guard_refusals": 0,
    "stage2": {
        "stage2_verdict": "admitted",
        "stage2_reason": "confirmed_by_second_frame",
        "candidate_cell_count": 64,
    },
}


def test_both_writers_project_the_same_source_dict_through_the_same_function():
    """THE UNIFICATION ITSELF. Not "both call something named hud_row_fields" -- the SAME object.

    The pre-fix state was two hand-rolled projections that happened to overlap on 9 of their keys.
    Identity of the function object is what makes "a field added for one harness is present on the
    other" true by construction instead of by memory.
    """

    harness = _load(_HARNESS, "arc_scored_path_lever_harness")
    exp = _load(_EXP5836, "experiment_5836_frontier_discipline_ab")
    assert harness.hud_row_fields is schema.hud_row_fields
    assert exp.hud_row_fields is schema.hud_row_fields
    # The historical spelling the harness re-exports must alias the canonical list, not copy it.
    assert harness.HUD_FLAT_ROW_KEYS is schema.HUD_ROW_KEYS


def test_both_writers_emit_the_identical_flat_key_set():
    """A writer that emits a SUBSET is the exact defect: exp5836 omitted the two shipped-side
    comparators, which is why its 1713 recorded rows cannot answer the lever-2 question at all.

    Asserted on the SET, so adding a key to one writer without the other fails here rather than
    silently producing rows that read None at half the call sites.
    """

    harness_src = _HARNESS.read_text(encoding="utf-8")
    exp_src = _EXP5836.read_text(encoding="utf-8")
    # Both writers must splat the shared projection rather than hand-listing keys.
    assert "row.update(hud_row_fields(hud))" in harness_src, (
        "the scored-path harness must project via the shared function"
    )
    assert "**hud_row_fields(hud_diag)" in exp_src, (
        "exp5836's main row must project via the shared function, not a hand-picked subset"
    )
    assert "**hud_row_fields(None)" in exp_src, (
        "exp5836's reference-solver and crash rows must emit the FULL key set (all-None) so an "
        "absent key can never be read as a measured zero"
    )
    fields = schema.hud_row_fields(REAL_R11L_DIAGNOSTICS)
    assert set(schema.HUD_ROW_KEYS) <= set(fields)
    # The deprecated alias must be present too, because exp5836's own three readers still use it.
    assert schema.NODE_INFLATION_ALIAS in fields
    assert fields[schema.NODE_INFLATION_ALIAS] == fields["node_inflation_vs_unique_frames"] == 1.108


def test_a_nested_row_and_a_flat_row_of_the_same_cell_score_identically():
    """THE DEFECT, REPRODUCED AS A TEST. One measurement, written in the two historical schemas,
    must produce the same answer through the consumer chokepoint.

    Before the fix this assertion failed in BOTH directions: the flat reader saw None on all 805
    nested rows, and the nested reader returned False on all 1713 flat rows.
    """

    analyzer = _load(_ANALYZER, "analyze_scored_path_lever_ab")
    flat = schema.hud_row_fields(REAL_R11L_DIAGNOSTICS)

    nested_row = {"game": "r11l", "seed": 1, "lever2_hud_fire": dict(REAL_R11L_DIAGNOSTICS)}
    flat_row = {"game": "r11l", "seed": 1, **flat}

    assert schema.backfill_hud_flat_fields(nested_row) == "nested_only"
    assert schema.backfill_hud_flat_fields(flat_row) == "flat_only"

    # Same verdict from the analyser, whichever schema the row arrived in.
    assert analyzer.recomputed_lever2_fired(nested_row) is True
    assert analyzer.recomputed_lever2_fired(flat_row) is True
    assert analyzer.fire_flag(nested_row, "lever2_fired") is True
    assert analyzer.fire_flag(flat_row, "lever2_fired") is True

    # And the same values at the FLAT address, which is what every non-lever reader uses.
    for key in ("hud_mask_resolved", "hud_mask_cell_count", "hud_mask_digest", "hud_mask_source"):
        assert nested_row[key] == flat_row[key], key
    # The stage-2 keys live one level deeper in the nested spelling and must still be lifted.
    assert nested_row["hud_mask_stage2_verdict"] == "admitted"
    # The name collision must resolve in BOTH directions, or the same quantity reads None on half
    # the corpus -- the identical defect, one field deeper.
    assert nested_row["node_inflation_vs_unique_frames"] == 1.108
    assert nested_row[schema.NODE_INFLATION_ALIAS] == 1.108
    legacy_short_name_only = {"game": "r11l", "seed": 1, schema.NODE_INFLATION_ALIAS: 1.108}
    schema.backfill_hud_flat_fields(legacy_short_name_only)
    assert legacy_short_name_only["node_inflation_vs_unique_frames"] == 1.108


def test_a_row_with_no_shipped_comparator_is_unscoreable_not_a_non_fire():
    """THE GUARD THAT WOULD HAVE CAUGHT THE REMAINING GAP. `hud_lever_fired` returns False for two
    structurally different reasons and only one of them is evidence.

    Measured: all 1713 recorded `results/cptb_20260726_cells/*.jsonl.gz` rows carry
    `hud_mask_digest` but no `hud_shipped_mask_digest`, because exp5836's pre-fix projection never
    wrote it. A denominator that pools them reports "the lever fired in 0 of 1713 cells" when the
    honest statement is "1713 cells cannot say" -- and no back-fill can recover a value that was
    never written, so only a re-run closes it.

    Note also what must NOT happen: treating the missing key as None would make `digest != None`
    true for EVERY resolved mask and arithmetically force "fired" on 1058 of those rows without a
    single shipped-side comparison ever having occurred.
    """

    cptb_shaped = {
        "hud_mask_resolved": True,
        "hud_mask_cell_count": 64,
        "hud_mask_digest": "fcbba0b6818499b6",
        "hud_mask_source": "edge_bar_detector_req5960_stage2_confirmed",
        # hud_shipped_mask_digest deliberately ABSENT -- this is the recorded exp5836 row shape.
    }
    assert schema.hud_lever_fired(cptb_shaped) is False
    assert schema.lever2_scoreable(cptb_shaped) is False, (
        "a row with no shipped-side comparator carries NO lever-2 evidence and must be excluded "
        "from the denominator, never counted as a non-fire"
    )

    row = dict(cptb_shaped)
    schema.backfill_hud_flat_fields(row)
    assert row["_hud_row_schema"] == "flat_only"
    assert row["_lever2_scoreable"] is False

    # The SAME row once the comparator is recorded IS scoreable -- and fires, because a mask
    # appearing where the shipped config had none is the lever's strongest possible firing.
    with_comparator = dict(cptb_shaped, hud_shipped_mask_digest=None)
    assert schema.lever2_scoreable(with_comparator) is True
    assert schema.hud_lever_fired(with_comparator) is True

    # A cell where the detector resolved nothing is scoreable and honestly did not fire -- that IS
    # evidence, and must not be pooled with the unscoreable rows above.
    resolved_nothing = {
        "hud_mask_resolved": False,
        "hud_mask_digest": None,
        "hud_shipped_mask_digest": None,
    }
    assert schema.lever2_scoreable(resolved_nothing) is True
    assert schema.hud_lever_fired(resolved_nothing) is False


def test_absent_and_errored_rows_carry_no_evidence_in_either_direction():
    """`None` is not `False`. An absent HUD block and a crashed diagnostics read must both be
    unscoreable, because a clean error-free zero from an unread channel is the dead-channel defect
    that once let a 72-97%-crashed control be published as a legitimate null."""

    absent = {"game": "synthetic", "seed": 1}
    assert schema.backfill_hud_flat_fields(absent) == "absent"
    assert absent["_lever2_scoreable"] is False

    errored = {"hud_diagnostics_error": "AttributeError:boom", "hud_mask_resolved": True}
    assert schema.hud_lever_fired(errored) is False
    assert schema.lever2_scoreable(errored) is False

    # The projection of "no explorer at all" is the full key set, all-None, with the witness False.
    empty = schema.hud_row_fields(None)
    assert set(schema.HUD_ROW_KEYS) <= set(empty)
    assert empty["hud_diagnostics_readable"] is False
    assert empty["lever2_fired"] is False
    assert schema.lever2_scoreable(empty) is False, (
        "an all-None projection records the absence explicitly, but it is still absence"
    )


def test_the_fire_predicate_has_exactly_one_implementation():
    """A recomputation that disagrees with the harness's stamp is the SIGNAL this analysis reports,
    so the recomputation must not itself be a second, independently-drifting copy of the rule."""

    analyzer = _load(_ANALYZER, "analyze_scored_path_lever_ab")
    harness = _load(_HARNESS, "arc_scored_path_lever_harness")
    assert harness.hud_lever_fired is schema.hud_lever_fired
    for hud in (
        REAL_R11L_DIAGNOSTICS,
        {"hud_mask_resolved": False},
        {"error": "boom"},
        {},
        dict(REAL_R11L_DIAGNOSTICS, hud_shipped_mask_digest="fcbba0b6818499b6"),
    ):
        row = {"game": "x", "seed": 1, **schema.hud_row_fields(hud)}
        assert analyzer.recomputed_lever2_fired(row) == schema.hud_lever_fired(hud)
    # Identical masks on both sides is a resolved-but-NOT-fired cell: real evidence, not a gap.
    identical = dict(REAL_R11L_DIAGNOSTICS, hud_shipped_mask_digest="fcbba0b6818499b6")
    assert schema.hud_lever_fired(identical) is False
    assert schema.lever2_scoreable(identical) is True

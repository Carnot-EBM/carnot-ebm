#!/usr/bin/env python3
"""Mechanical `ast` classification of an induced `is_level_complete` into the three shapes the
2026-08-02 goal-evidence question is about: DECLINED, TROPE, GROUNDED.

WHY A SHAPE CLASSIFIER IS THE PRIMARY AND THE GOAL GATE IS NOT. `plan_found` is an EXACT
function of the goal gate's kind -- 0 mismatches in 138 engines -- so grading a GOAL
intervention with it grades the gate using the gate. Predicate SHAPE is decidable from the
syntax tree alone: it needs no environment, no bounded search, no win, and it is independent of
the gate. It is also a direct read on the defect that motivated the run (34 of 71 goal-failing
engines return `False` unconditionally, four of them saying in their own docstring that they
were told nothing about the win condition).

NO INDUCED CODE IS EXECUTED HERE, deliberately and for the same reason the 2026-08-01 anatomy
pass gave: the corpus contains a candidate with a measured non-terminating loop, and the
standing project rule is that induced code runs in a killable subprocess or not at all. Every
property this module reports is syntactic.

THE CLUSTER LAYER IS REUSED, NOT RE-DERIVED. `results/arc_goal_predicate_anatomy_20260801/
anatomy.py:classify` already partitions predicates into A_DECLINED / D_NO_PREDICATE /
I_CONSTANT_TRUE / C_UNIFORMITY / B_COLOUR_ELIMINATION / E_FIXED_BAND / F_OBJECT_POSITION /
G_CONNECTIVITY / H_OTHER, and the corpus taxonomy this experiment is testing was written in
those terms. Re-implementing the partition would let the two drift, and any drift would look
like a treatment effect. So this module imports that `classify` and maps its output up, adding
exactly ONE thing it cannot know: whether the predicate's literals correspond to something the
agent ACTUALLY OBSERVED.

THE GROUNDING CHECK IS THE ONE NEW IDEA. "GROUNDED" in the brief is "names a region, cell or
object appearing in the observed transitions" -- strictly more than "mentions a literal". A
predicate that tests `grid[7, 3] == 4` is grounded only if row 7 / col 3 / colour 4 is something
the deltas the model was shown actually contain. This matters because the failure mode under
test is a model inventing a plausible-sounding ARC trope in the absence of evidence; a predicate
citing coordinates that appear nowhere in its own observations is that same failure wearing a
subscript. The check uses ONLY the agent's own transitions, so it is exactly as computable on a
game nobody has ever solved.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

REPO = Path(__file__).resolve().parents[2]
_ANATOMY = REPO / "results" / "arc_goal_predicate_anatomy_20260801" / "anatomy.py"

_spec = importlib.util.spec_from_file_location("_goal_anatomy", _ANATOMY)
assert _spec is not None and _spec.loader is not None
anatomy = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(anatomy)

# The three PRIMARY shapes plus the residual. They are reported as four separate rates and are
# NOT assumed to sum to 1 by any consumer -- OTHER is a real bucket, not a rounding error.
DECLINED_CLUSTERS = {"A_DECLINED", "D_NO_PREDICATE"}
TROPE_CLUSTERS = {"C_UNIFORMITY", "B_COLOUR_ELIMINATION"}
# Clusters that CAN be grounded: they name something (a region, a cell, an object) rather than
# making a whole-board claim. Whether they ARE grounded depends on the transitions.
GROUNDABLE_CLUSTERS = {"E_FIXED_BAND", "F_OBJECT_POSITION", "G_CONNECTIVITY", "H_OTHER"}


def last_goal_def(src: str) -> tuple[ast.FunctionDef | None, int, str | None]:
    """(the definition Python would BIND, how many top-level ones there are, error).

    The LAST top-level `def is_level_complete` is the one that runs -- earlier ones are dead.
    That is the whole reason the dedup knob exists, so analysing anything else would analyse
    code the planner never calls.
    """
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        return None, 0, f"SyntaxError:{exc.msg}"
    defs = [
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "is_level_complete"
    ]
    if not defs:
        return None, 0, "no_is_level_complete_def"
    return defs[-1], len(defs), None


def _int_literals(fn: ast.FunctionDef) -> set[int]:
    """Every small non-negative integer literal in the predicate.

    Deliberately crude and deliberately INCLUSIVE of both axes: a row index, a column index and
    a colour are all small ints and the AST cannot always tell them apart (`grid[r] == 4` and
    `grid[4] == r` differ only in position). Being inclusive makes the grounding test EASIER to
    pass, which biases against the hypothesis this run is testing -- the safe direction. Values
    above 63 are dropped: ARC grids are at most 64x64 and the palette is 16 wide, so a larger
    literal is arithmetic (a count, a threshold), not a coordinate or a colour.
    """
    out: set[int] = set()
    for n in ast.walk(fn):
        if (
            isinstance(n, ast.Constant)
            and isinstance(n.value, int)
            and not isinstance(n.value, bool)
            and 0 <= n.value <= 63
        ):
            out.add(int(n.value))
    return out


def observed_vocabulary(transitions: Sequence[Any]) -> dict[str, set[int]]:
    """What the agent ACTUALLY SAW, as the sets a predicate's literals can be checked against.

    Three sets, all read from the deltas the induce prompt renders:
      rows/cols -- the coordinates of cells that CHANGED in some observed transition
      colours   -- the values those cells held before or after the change

    Only CHANGED cells count, not the whole board. A predicate citing a colour that is simply
    present somewhere on a static background has not demonstrated it read the deltas; the claim
    "names something appearing in the observed transitions" is about the transitions.
    """
    import numpy as np

    rows: set[int] = set()
    cols: set[int] = set()
    colours: set[int] = set()
    for t in transitions:
        before = getattr(t, "grid", None)
        after = getattr(t, "next_grid", None)
        if before is None or after is None:
            continue
        a = np.asarray(before)
        b = np.asarray(after)
        if a.shape != b.shape:
            # A resize is a change of everything; record the palettes and no coordinates, since
            # coordinates are not comparable across shapes.
            colours |= {int(v) for v in np.unique(a)} | {int(v) for v in np.unique(b)}
            continue
        rr, cc = np.nonzero(a != b)
        for r, c in zip(rr.tolist(), cc.tolist(), strict=True):
            rows.add(int(r))
            cols.add(int(c))
            colours.add(int(a[r, c]))
            colours.add(int(b[r, c]))
    return {"rows": rows, "cols": cols, "colours": colours}


def classify_predicate(src: str, transitions: Sequence[Any] | None = None) -> dict[str, Any]:
    """-> {shape, cluster, cluster_note, n_defs, grounding_hits, error}.

    `shape` is one of DECLINED / TROPE / GROUNDED / OTHER, or None when the predicate could not
    be recovered at all (a MISSING OBSERVATION -- the caller must exclude it and count it per
    arm, never score it 0).
    """
    fn, n_defs, err = last_goal_def(src)
    rec: dict[str, Any] = {"n_defs": n_defs, "error": err}
    if fn is None:
        rec["shape"] = None
        return rec
    cluster, note = anatomy.classify(fn)
    rec["cluster"] = cluster
    rec["cluster_note"] = note
    rec["docstring_says_no_win_state"] = "docstring" in note

    if cluster in DECLINED_CLUSTERS:
        rec["shape"] = "DECLINED"
        return rec
    if cluster in TROPE_CLUSTERS:
        rec["shape"] = "TROPE"
        return rec
    if cluster == "I_CONSTANT_TRUE":
        # Neither a decline nor a trope nor grounded: degenerate in the OTHER direction. Kept
        # visible rather than folded into OTHER silently, because the shipped goal gate REJECTS
        # constant-True, so an arm that produced more of them is not improving.
        rec["shape"] = "OTHER"
        rec["other_kind"] = "constant_true"
        return rec

    lits = _int_literals(fn)
    rec["int_literals"] = sorted(lits)
    if transitions is None:
        rec["shape"] = "OTHER"
        rec["other_kind"] = "groundable_but_no_transitions_supplied"
        return rec
    vocab = observed_vocabulary(transitions)
    hits = {
        "rows": sorted(lits & vocab["rows"]),
        "cols": sorted(lits & vocab["cols"]),
        "colours": sorted(lits & vocab["colours"]),
    }
    rec["grounding_hits"] = hits
    grounded = bool(hits["rows"] or hits["cols"] or hits["colours"])
    if grounded and cluster in GROUNDABLE_CLUSTERS:
        rec["shape"] = "GROUNDED"
        return rec
    rec["shape"] = "OTHER"
    rec["other_kind"] = "names_nothing_observed" if not grounded else f"cluster_{cluster}"
    return rec


def shape_counts(records: Iterable[dict]) -> dict[str, int]:
    out = {"DECLINED": 0, "TROPE": 0, "GROUNDED": 0, "OTHER": 0, "MISSING": 0}
    for r in records:
        s = r.get("shape")
        out["MISSING" if s is None else s] += 1
    return out

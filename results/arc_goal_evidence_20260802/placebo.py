#!/usr/bin/env python3
"""DISCRIMINANCE (placebo) test for the GROUNDED label used by the goal-evidence A/B.

WHY THIS EXISTS. The A/B's SECONDARY_grounded_rate is defined as "the predicate names a
literal that appears in the agent's observed deltas". The artifact already discloses the
obvious circularity -- the treatment prompt IS those deltas, so a model that copies a row
index out of its prompt scores GROUNDED without understanding anything. That disclosure is
necessary but not sufficient, because it still assumes the label at least MEASURES the
agent's own transitions. This script tests that assumption directly and it does not hold.

THE TEST. Re-score every captured predicate against a DIFFERENT game's observed transitions.
A label that carries evidence about THIS game's deltas must mostly break when handed another
game's deltas. If it survives, the label is reporting the SYNTAX of the predicate (does it
mention a small integer at all) rather than its grounding in what the agent saw.

TWO CONSTRUCTED COUNTEREXAMPLES, both behaviour-preserving, both run against real captured
cells rather than synthetic fixtures:

  1. `return False`  ->  `return False and grid[0, 0] == 4`
     Python short-circuits, so the predicate is bit-identically the same function. Every
     real DECLINED cell flips to GROUNDED. This exactly INVERTS the run's reported S2-B/S2-C
     result (declined 1.0 / grounded 0.0 becomes declined 0.0 / grounded 1.0) from a model
     that got strictly no better.

  2. Alpha-rename a local, or swap the operands of an equality.
     TROPE is a hardcoded 5-substring match over `ast.unparse` (see anatomy.py's
     C_UNIFORMITY rule), so `len(non_zero_colors) == 1` -> `len(nzc) == 1` and
     `np.all(grid == grid[0, 0])` -> `np.all(grid[0, 0] == grid)` both escape the trope
     match into H_OTHER, where the grounding check then promotes them to GROUNDED.

NO INDUCED CODE IS EXECUTED. `classify.classify_predicate` is pure AST inspection (see
anatomy.py's module docstring) and the mutations here are textual, so nothing in a captured
completion is ever run. That is deliberate: this file reads adversarial LLM output.

WHAT THIS DOES NOT CLAIM. It does not say the predicates are bad, and it does not say the
PRIMARY (DECLINED) or TROPE contrasts are affected -- a constant-False predicate references
nothing and a whole-board uniformity claim names nothing observed, so neither can be
manufactured by prompt content. It says the GROUNDED label alone must be read as a syntax
label, and that the run's grounded rates must not be quoted as evidence-use rates.

Usage:  python3 results/arc_goal_evidence_20260802/placebo.py
        GEV_WINDOWS=/path/to/windows.pkl python3 .../placebo.py
"""

from __future__ import annotations

import json
import os
import pickle
import re
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "out"

# Point the e3 store at a scratch directory BEFORE importing anything from carnot. Rebuilding
# the windows walks the offline arcade, and `results/arc_e3/**` is tracked read-only EVIDENCE
# whose only runtime guard (`_guard_engine_write`) is pytest-scoped -- a measurement driver is
# exactly the caller nothing protects.
os.environ.setdefault(
    "CARNOT_ARC_E3_DIR",
    str(Path(os.environ.get("TMPDIR", "/tmp")) / "carnot_placebo_e3"),
)

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "python"))

import classify as gcl  # noqa: E402

WINDOWS_PKL = Path(
    os.environ.get("GEV_WINDOWS")
    or "/tmp/claude-1000/-home-ianblenke-github-com-ianblenke-carnot/"
    "87d32f9e-547c-4832-8fd3-2cabb283bc83/scratchpad/goalev/windows.pkl"
)

ROSTER = [
    "ar25",
    "bp35",
    "cd82",
    "cn04",
    "g50t",
    "ka59",
    "lf52",
    "lp85",
    "ls20",
    "m0r0",
    "re86",
    "s5i5",
    "sb26",
    "sc25",
    "sk48",
    "su15",
    "tn36",
    "tr87",
    "tu93",
    "wa30",
]


def load_windows() -> dict[str, tuple]:
    """The same (shown, heldout, cell) windows every arm of the A/B was scored against.

    Read from the cached pickle when it is still on disk so this reproduces the exact inputs;
    rebuilt from the recorded frames otherwise. Rebuilding is deterministic but slow (one game
    took ~200s), which is why the run cached it in the first place.
    """
    if WINDOWS_PKL.exists():
        return pickle.loads(WINDOWS_PKL.read_bytes())
    from carnot.agentic import arc_actions_to_progress as atp
    from carnot.agentic import arc_world_model_trust_energy as wmte

    out: dict[str, tuple] = {}
    for g in ROSTER:
        w = atp.build_progress_window(g)
        if w is None:
            continue
        win, _full, cell = w
        shown, held = wmte._split_prefix_heldout(list(win))  # noqa: SLF001
        out[g] = (shown, held, int(cell))
    return out


def captured_cells() -> list[tuple[str, str, str]]:
    """(cell name, home game, source) for every predicate this run actually recovered.

    The two stage-1 cells that returned no parseable predicate have no file and are therefore
    absent here -- they are a MISSING observation, not a zero, exactly as the analysis treats
    them.
    """
    cells = []
    for sub in ("s1_cells", "s2_cells"):
        for p in sorted((OUT / sub).glob("*.py.txt")):
            cells.append((p.name, p.name.split("__")[0], p.read_text()))
    return cells


def accepting_sets(windows: dict[str, tuple]) -> dict[str, set[int]]:
    """Per game, the union of observed rows / cols / colours clipped to the literal space.

    This is the set a predicate's integer literals are intersected against. Its SIZE is the
    headline number of this script: the grounding test is only informative if the accepting
    set is a small part of the space a predicate's literals can fall in.
    """
    out: dict[str, set[int]] = {}
    for g, (shown, _h, _c) in windows.items():
        v = gcl.observed_vocabulary(shown)
        out[g] = {x for x in (v["rows"] | v["cols"] | v["colours"]) if 0 <= x <= 63}
    return out


def shape_of(src: str, transitions: Any) -> str | None:
    return gcl.classify_predicate(src, transitions).get("shape")


def main() -> int:
    windows = load_windows()
    cells = captured_cells()
    acc = accepting_sets(windows)
    games = sorted(acc)

    sizes = {g: len(acc[g]) for g in games}
    literal_space = 64  # `_int_literals` keeps 0..63 and drops anything larger as arithmetic

    # ---- 1. how big is the accepting set, i.e. how easy is it to score GROUNDED by luck ----
    accepting = {
        "literal_space": literal_space,
        "per_game_accepting_set_size": sizes,
        "mean_fraction_of_literal_space": round(
            sum(sizes.values()) / len(sizes) / literal_space, 4
        ),
        "min_fraction": round(min(sizes.values()) / literal_space, 4),
        "max_fraction": round(max(sizes.values()) / literal_space, 4),
        "min_game": min(sizes, key=lambda g: sizes[g]),
        "max_game": max(sizes, key=lambda g: sizes[g]),
        "literal_0_grounds_on_n_games": sum(1 for g in games if 0 in acc[g]),
        "n_games": len(games),
        "reading": "a predicate's literals are intersected against this set. At a mean 66% of "
        "the literal space, naming ANY small integer is close to a coin flip in the "
        "grounded direction before the model has understood anything.",
    }

    # ---- 2. the placebo proper: does the label survive a DIFFERENT game's transitions? ----
    per_cell = []
    preserved = compared = 0
    for name, home, src in cells:
        native = shape_of(src, windows[home][0])
        foreign = {g: shape_of(src, windows[g][0]) for g in games if g != home}
        same = sum(1 for s in foreign.values() if s == native)
        preserved += same
        compared += len(foreign)
        per_cell.append(
            {
                "cell": name,
                "home_game": home,
                "native_shape": native,
                "n_foreign_games": len(foreign),
                "n_foreign_preserving_the_label": same,
                "n_foreign_scoring_GROUNDED": sum(1 for s in foreign.values() if s == "GROUNDED"),
            }
        )
    placebo = {
        "what": "each captured predicate re-scored against every OTHER game's observed "
        "transitions. A label that carries evidence about THIS game's deltas should "
        "mostly break; one that survives is reporting syntax.",
        "n_cells": len(cells),
        "n_comparisons": compared,
        "label_preserved_under_a_foreign_games_transitions": round(preserved / compared, 4),
        "grounded_cells": [c for c in per_cell if c["native_shape"] == "GROUNDED"],
        "per_cell": per_cell,
    }

    # ---- 3. two behaviour-preserving mutations on REAL cells --------------------------------
    decl_flips = []
    for name, home, src in cells:
        if shape_of(src, windows[home][0]) != "DECLINED":
            continue
        mut = re.sub(r"return\s+False\b", "return False and grid[0, 0] == 4", src)
        if mut == src:
            continue  # a D_NO_PREDICATE cell with no `return False` to short-circuit
        decl_flips.append({"cell": name, "was": "DECLINED", "now": shape_of(mut, windows[home][0])})
    trope_flips = []
    for name, home, src in cells:
        if shape_of(src, windows[home][0]) != "TROPE":
            continue
        rename = src.replace("non_zero_colors", "nzc").replace("non_zero_colours", "nzc")
        swap = re.sub(
            r"np\.all\(\s*grid\s*==\s*grid\[0,\s*0\]\s*\)", "np.all(grid[0, 0] == grid)", src
        )
        row: dict[str, Any] = {"cell": name, "was": "TROPE"}
        if rename != src:
            row["alpha_rename_local"] = shape_of(rename, windows[home][0])
        if swap != src:
            row["swap_equality_operands"] = shape_of(swap, windows[home][0])
        trope_flips.append(row)
    mutations = {
        "short_circuit_on_DECLINED": {
            "mutation": "`return False` -> `return False and grid[0, 0] == 4`",
            "why_it_is_behaviour_preserving": "Python short-circuits `False and X`, so the "
            "function returns False for every input exactly as before. No induced code is "
            "executed to establish this; it is a language guarantee.",
            "cells": decl_flips,
            "consequence": "the reported S2-B / S2-C result (declined 1.0, grounded 0.0) "
            "inverts to (declined 0.0, grounded 1.0) under a rewrite that changes nothing "
            "the predicate does.",
        },
        "paraphrase_on_TROPE": {
            "mutation": "alpha-rename a local, or swap the operands of an equality",
            "why_it_is_behaviour_preserving": "renaming a local and commuting `==` are both "
            "semantics-preserving; TROPE is a 5-substring match over `ast.unparse`, so the "
            "paraphrase escapes into H_OTHER and the grounding check then promotes it.",
            "cells": trope_flips,
        },
    }

    doc = {
        "what_this_is": "a DISCRIMINANCE (placebo) test of the GROUNDED label, run AFTER the "
        "A/B on the A/B's own captured cells. Post-hoc and labelled as such.",
        "verdict": "GROUNDED IS A SYNTAX LABEL, NOT AN EVIDENCE LABEL. It must not be quoted "
        "as an evidence-use rate. DECLINED (the PRIMARY) and TROPE are NOT affected: a "
        "constant-False predicate references nothing and a whole-board claim names nothing "
        "observed, so no prompt content can manufacture either.",
        "accepting_set": accepting,
        "placebo": placebo,
        "behaviour_preserving_mutations": mutations,
        "no_induced_code_executed": True,
    }
    (OUT / "grounded_placebo.json").write_text(json.dumps(doc, indent=2) + "\n")
    print(json.dumps({k: v for k, v in doc.items() if k in ("verdict",)}, indent=2))
    print("accepting set mean fraction:", accepting["mean_fraction_of_literal_space"])
    print(
        "label preserved under foreign transitions:",
        placebo["label_preserved_under_a_foreign_games_transitions"],
    )
    print("DECLINED->? under short-circuit:", [(c["cell"], c["now"]) for c in decl_flips])
    print("wrote", OUT / "grounded_placebo.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

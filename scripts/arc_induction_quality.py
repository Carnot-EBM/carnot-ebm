#!/usr/bin/env python3
"""Score the world models the live agent actually emitted, without needing a valid run.

WHY THIS EXISTS. For 24 hours every live-agent measurement came back
`llm_on_row_valid: false`, and that was read as "we learned nothing". It gated
five successive runs' worth of conclusions. It should not have: `llm_on_row_valid`
gates COMPARISONS between arms, because an arm whose generator died cannot be
compared to one whose generator lived. It says nothing about whether the model
induced anything worthwhile, and the evidence for THAT was on disk the whole time
— `world_model.py` under the run's `CARNOT_ARC_E3_DIR`, written the moment an
induction succeeds, surviving whatever kills the harness later.

The greenrun of 2026-08-23 is the case in point: harness died on SIGSEGV after
1h54m with ZERO rows, and had already emitted a 229-line ar25 world model with
connected-component labelling and a goal predicate over induced structure.

WHAT THIS MEASURES, precisely: whether an emitted model is SUBSTANTIVE or
DEGENERATE. Branch count, function count, and the shape of the goal predicate
separate "the model wrote a real perception routine" from "the model returned the
grid unchanged" or "the goal predicate is a memorised literal".

WHAT THIS DOES NOT MEASURE, and must never be read as: CORRECTNESS. A model can
be elaborate and wrong. Held-out transition accuracy is the only thing that speaks
to correctness, and it needs a completed run. Treating a branch count as evidence
of a working agent would be exactly the "field names lie" trap this project keeps
paying for — a number that sounds like quality standing in for quality.

So: use this to answer "is the generator producing real work at all", which is
cheap and answerable now. Use rows to answer "is arm A better than arm B", which
is expensive and needs a run that survives.

Read-only. Touches nothing.
"""

from __future__ import annotations

import argparse
import ast
import os
import re
from pathlib import Path

DEFAULT_ROOT = (
    Path(os.environ.get("CLAUDE_JOB_DIR", "/home/ianblenke/.claude/jobs/ad0c053d")) / "tmp"
)

# A goal predicate that consults structure the model itself derived (components,
# labels, masks, objects) is doing perception. One that only compares against a
# stored literal is memorising. Neither list is exhaustive; both are reported
# rather than thresholded, because a threshold here would be fitted to the
# handful of models seen so far.
_STRUCTURE_TERMS = (
    "component",
    "comps",
    "label",
    "mask",
    "object",
    "region",
    "connected",
    "flood",
    "neighbou",
    "neighbor",
    "adjacen",
)
_GOAL_FN = re.compile(r"def\s+(is_level_complete|is_win|is_goal|goal_reached)\b")


def score_model(path: Path) -> dict:
    """Structural scoring of one emitted world_model.py. Never imports it."""
    try:
        src = path.read_text(encoding="utf-8")
    except OSError as exc:
        return {"path": str(path), "error": f"unreadable: {exc}"}

    out: dict = {
        "path": str(path),
        "game": path.parent.name,
        "bytes": len(src.encode("utf-8")),
        "lines": src.count("\n") + 1,
    }

    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        # A model that does not parse is a hard negative and worth surfacing
        # loudly: it means the generator emitted something the runtime cannot use.
        out["error"] = f"SYNTAX ERROR line {exc.lineno}: {exc.msg}"
        out["degenerate"] = True
        return out

    fns = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    branches = [
        n
        for n in ast.walk(tree)
        if isinstance(n, (ast.If, ast.For, ast.While, ast.IfExp, ast.Compare))
    ]
    out["functions"] = len(fns)
    out["branches"] = len(branches)

    # Degeneracy tells. Each is a way of "answering" without modelling anything.
    tells: list[str] = []
    for fn in fns:
        body = [
            s
            for s in fn.body
            if not isinstance(s, ast.Expr)
            or not isinstance(getattr(s, "value", None), ast.Constant)
        ]
        if len(body) == 1:
            only = body[0]
            if isinstance(only, ast.Pass):
                tells.append(f"{fn.name}: body is `pass`")
            elif isinstance(only, ast.Return):
                v = only.value
                if isinstance(v, ast.Name) and v.id in ("grid", "state"):
                    tells.append(f"{fn.name}: returns its input unchanged")
                elif isinstance(v, ast.Constant) and isinstance(v.value, bool):
                    tells.append(f"{fn.name}: returns constant {v.value}")
    out["degeneracy_tells"] = tells
    out["degenerate"] = bool(tells)

    # Does the goal predicate reference structure the model derived?
    goal_fn = next((f for f in fns if _GOAL_FN.match(f"def {f.name}(")), None)
    if goal_fn is None:
        out["goal_predicate"] = "ABSENT"
    else:
        seg = ast.get_source_segment(src, goal_fn) or ""
        hits = sorted({t for t in _STRUCTURE_TERMS if t in seg.lower()})
        out["goal_predicate"] = "structural" if hits else "flat"
        out["goal_structure_terms"] = hits
        out["goal_branches"] = sum(
            1
            for n in ast.walk(goal_fn)
            if isinstance(n, (ast.If, ast.For, ast.While, ast.IfExp, ast.Compare))
        )
    return out


def find_models(roots: list[Path]) -> list[Path]:
    found: list[Path] = []
    for root in roots:
        if root.is_dir():
            found.extend(sorted(root.rglob("world_model.py")))
    return found


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roots", nargs="*", default=[str(DEFAULT_ROOT)])
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    models = find_models([Path(r) for r in args.roots])
    if not models:
        print("no world_model.py found under", args.roots)
        return 1

    scored = [score_model(p) for p in models]

    if args.json:
        import json

        print(json.dumps(scored, indent=2))
        return 0

    print("")
    print("  ARC INDUCTION QUALITY — emitted world models (structure, NOT correctness)")
    print("  " + "-" * 68)
    print(f"  {'run/game':28} {'lines':>6} {'fns':>4} {'br':>4}  {'goal':10} tells")
    for s in scored:
        run = Path(s["path"]).parts[-4] if len(Path(s["path"]).parts) >= 4 else "?"
        label = f"{run}/{s.get('game', '?')}"[:28]
        if s.get("error"):
            print(f"  {label:28} {s['error']}")
            continue
        tells = "; ".join(s["degeneracy_tells"]) or "-"
        print(
            f"  {label:28} {s['lines']:>6} {s['functions']:>4} {s['branches']:>4}"
            f"  {s.get('goal_predicate', '?'):10} {tells}"
        )
    n_deg = sum(1 for s in scored if s.get("degenerate"))
    n_struct = sum(1 for s in scored if s.get("goal_predicate") == "structural")
    print("")
    print(
        f"  {len(scored)} model(s); {n_deg} degenerate; {n_struct} with a structural goal predicate"
    )
    print("  Structure is not correctness. Held-out transition accuracy needs a run that survives.")
    print("")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

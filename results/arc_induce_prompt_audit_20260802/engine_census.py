#!/usr/bin/env python3
"""Census the SHAPE of the engines currently on disk. READ ONLY.

`results/arc_e3` is EVIDENCE per CLAUDE.md: read, never write. This script opens
`world_model.py` under each game directory and answers one question by static analysis --
does `engine()` have any path that returns something other than its input grid?

WHY STATIC AND NOT EXECUTED. Executing these is what wedged a prior session for 13 minutes
(`arc_engine_static_validation.engine_changes_anything` is documented as unbounded, and one
ft09 candidate never returns). The question here is about the CODE THE MODEL WROTE, which is a
property of the text, so the text is what is read.

WHAT "IDENTITY" MEANS HERE, precisely. Two conditions, and the second is the one a first
draft of this script got wrong. (1) every `return` in the top-level `engine` returns the bare
parameter, a copy of it, or a no-keyword call over it; AND (2) the function does not write
THROUGH the parameter. su15 is the counterexample that forced (2): it does
`grid[py, px] = 15` and then `return grid`, so every return is param-like while the function
genuinely changes the grid. Collapsing those two cases -- as the first version did -- reported
su15 as identity, which is false. `mutates_param_in_place` is therefore a separate verdict and
is NOT counted as identity.

`empty_branch_rate` = param-like returns / total returns. This is the CONTINUOUS measure and
the more informative one: ft09 scores 12/12 not because it is a stub but because it builds a
full click-dispatch with eight colour cases and writes `return grid` in every consequent. A
binary identity flag cannot see that shape at all.
"""

import ast
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
E3 = REPO / "results" / "arc_e3"
OUT = Path(__file__).resolve().parent / "out" / "engine_census.json"


def classify(src: str) -> dict:
    try:
        tree = ast.parse(src)
    except SyntaxError as exc:
        return {"verdict": "unparseable", "detail": str(exc)[:120]}
    fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "engine"), None)
    if fn is None:
        return {"verdict": "no_top_level_engine"}
    param = fn.args.args[0].arg if fn.args.args else None
    returns = [n for n in ast.walk(fn) if isinstance(n, ast.Return)]
    if not returns:
        return {"verdict": "no_return", "n_returns": 0}

    # Any assignment / augmented assignment / subscript store anywhere in the body means the
    # function is at least ATTEMPTING to build a different grid.
    mutates = any(isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign)) for n in ast.walk(fn))

    def returns_param_like(r: ast.Return) -> bool:
        v = r.value
        if v is None:
            return False
        if isinstance(v, ast.Name) and v.id == param:
            return True
        # grid.copy(), np.array(grid), np.asarray(grid) -- a copy is still identity
        if isinstance(v, ast.Call):
            args = [a for a in v.args if isinstance(a, ast.Name)]
            if (
                isinstance(v.func, ast.Attribute)
                and isinstance(v.func.value, ast.Name)
                and v.func.value.id == param
                and v.func.attr in {"copy", "view"}
            ):
                return True
            if any(a.id == param for a in args) and not v.keywords:
                return True
        return False

    # IN-PLACE MUTATION OF THE PARAMETER is the case that makes "returns the parameter on every
    # path" NOT identity. su15 does `grid[py, px] = 15` then `return grid`: every return is the
    # bare parameter, yet the function genuinely changes the grid. Separating these two is
    # load-bearing -- collapsing them would have reported su15 as identity, which is false.
    inplace = False
    for n in ast.walk(fn):
        if isinstance(n, (ast.Assign, ast.AugAssign)):
            targets = n.targets if isinstance(n, ast.Assign) else [n.target]
            for t in targets:
                if isinstance(t, ast.Subscript):
                    base = t.value
                    while isinstance(base, (ast.Subscript, ast.Attribute)):
                        base = base.value
                    if isinstance(base, ast.Name) and base.id == param:
                        inplace = True
        # grid.fill(0) / grid.itemset(...) / np.place(grid, ...) are also in-place writers.
        if (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and isinstance(n.func.value, ast.Name)
            and n.func.value.id == param
            and n.func.attr in {"fill", "itemset", "put", "sort", "resize"}
        ):
            inplace = True

    all_param = all(returns_param_like(r) for r in returns)
    n_lines = len(src.splitlines())
    if all_param and inplace:
        # Returns the parameter everywhere, but writes through it: behaviourally NOT identity.
        v = "mutates_param_in_place"
    elif all_param and not mutates:
        v = "identity_no_work"
    elif all_param and mutates:
        # Assigns only to locals, then returns the untouched parameter: the work is DEAD.
        v = "identity_with_dead_work"
    else:
        v = "has_a_non_param_return"
    return {
        "verdict": v,
        "behaviourally_identity": v in {"identity_no_work", "identity_with_dead_work"},
        "n_returns": len(returns),
        "n_returns_param_like": sum(1 for r in returns if returns_param_like(r)),
        "assigns_anything": mutates,
        "mutates_param_in_place": inplace,
        "n_lines": n_lines,
        "param": param,
    }


rows = []
for d in sorted(E3.iterdir()):
    wm = d / "world_model.py"
    if not d.is_dir() or not wm.exists():
        continue
    src = wm.read_text(errors="replace")
    rec = {"game": d.name, "bytes": len(src)}
    rec.update(classify(src))
    rows.append(rec)

for r in rows:
    nr, npl = r.get("n_returns") or 0, r.get("n_returns_param_like") or 0
    r["empty_branch_rate"] = round(npl / nr, 4) if nr else None

verdicts: dict[str, int] = {}
for r in rows:
    verdicts[r["verdict"]] = verdicts.get(r["verdict"], 0) + 1

out = {
    "source": str(E3),
    "read_only": True,
    "n_engines_on_disk": len(rows),
    "verdict_counts": verdicts,
    "behaviourally_identity_games": sorted(
        r["game"] for r in rows if r.get("behaviourally_identity")
    ),
    "n_behaviourally_identity": sum(1 for r in rows if r.get("behaviourally_identity")),
    "empty_branch_rate": {
        "definition": "param-like returns / total returns in engine(); 1.0 = every branch "
        "returns the input untouched",
        "per_game": {r["game"]: r["empty_branch_rate"] for r in rows},
        "n_at_1_0": sum(1 for r in rows if r.get("empty_branch_rate") == 1.0),
        "n_at_0_0": sum(1 for r in rows if r.get("empty_branch_rate") == 0.0),
        "n_engines": len(rows),
        "median": sorted(
            r["empty_branch_rate"] for r in rows if r.get("empty_branch_rate") is not None
        )[len(rows) // 2],
    },
    "rows": rows,
}
OUT.write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))

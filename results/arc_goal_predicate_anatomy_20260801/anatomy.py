#!/usr/bin/env python3
"""Extract every induced `is_level_complete` from the two frozen engine corpora and cluster the
predicates by WHAT THEY ACTUALLY CHECK.

PURE AST INSPECTION. No induced code is executed anywhere in this module, deliberately: the
corpus contains a candidate with a measured non-terminating loop (ft09 k5), and the repo's
standing rule is that induced code runs in a killable subprocess or not at all. Every property
reported here -- "returns a constant", "raises NameError", "decides on a literal grid region" --
is decidable from the syntax tree, so nothing is lost by not running it.

WHY THE *LAST* DEFINITION IS THE ONE ANALYSED. 22 of the 114 object-perception engines define
`is_level_complete` TWICE. Python binds the last top-level definition, so that is the predicate
the planner actually calls; the earlier one is dead. Those 22 are exactly the 22 engines carrying
`_combine_world_model`'s double-`import numpy as np` signature -- the split-induce fallback, which
concatenates an engine block that already contained a predicate with a separately-generated
goal-only block. The dead first definition is the one written with the transitions in context.
"""

from __future__ import annotations

import ast
import builtins
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OBJPERC = REPO / "results" / "arc_object_perception_ab_change_fidelity_20260801"
BESTOFN = REPO / "results" / "arc_induce_bestofn_20260731"
METRIC_VALIDITY = REPO / "results" / "arc_metric_validity_20260801" / "analysis.json"

# Names a predicate may reference without it being a NameError at call time.
SAFE = set(dir(builtins)) | {"np", "numpy", "grid", "math", "collections", "deque"}


def extract_python(text: str) -> str:
    """Pull the first python code block. Byte-identical to
    `arc_executable_world_model._extract_python`, which is how the best-of-N completions were
    turned into engines -- re-deriving them any other way would analyse code that never ran."""
    if "```python" in text:
        text = text.split("```python", 1)[1]
    if "```" in text:
        text = text.split("```", 1)[0]
    return text.strip()


def source_for(row: dict) -> tuple[str | None, str]:
    """Resolve a scored row back to the exact engine text that was scored."""
    cell = row["cell"]
    if row["corpus"] == "objperc":
        cands = sorted((OBJPERC / "engines" / cell).rglob("world_model.py"))
        if not cands:
            return None, "no_world_model_py"
        return cands[0].read_text(errors="replace"), str(cands[0].relative_to(REPO))
    game, k = cell.split("__k")
    p = BESTOFN / "harness" / "bon" / "gpu1" / f"{game}_k{k}.txt"
    if not p.exists():
        return None, "no_completion_txt"
    return extract_python(p.read_text(errors="replace")), str(p.relative_to(REPO))


def _free_names(fn: ast.FunctionDef) -> set[str]:
    """Names LOADED but never bound anywhere in the function. A non-SAFE free name means the
    predicate raises NameError the first time the planner calls it, which the goal gate reports
    as an error rather than as a goal."""
    bound = {a.arg for a in fn.args.args} | {a.arg for a in getattr(fn.args, "kwonlyargs", [])}
    loaded: set[str] = set()
    for n in ast.walk(fn):
        if isinstance(n, ast.Name):
            (bound if isinstance(n.ctx, (ast.Store, ast.Del)) else loaded).add(n.id)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for al in n.names:
                bound.add((al.asname or al.name).split(".")[0])
        elif (isinstance(n, ast.ExceptHandler) and n.name) or isinstance(
            n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            bound.add(n.name)
    return {x for x in loaded - bound if x not in SAFE}


def _const_returns(fn: ast.FunctionDef) -> tuple[bool, set]:
    """(every return is a literal, the set of literal values returned)."""
    rs = [n for n in ast.walk(fn) if isinstance(n, ast.Return)]
    vals: set = set()
    for r in rs:
        if r.value is None:
            vals.add(None)
        elif isinstance(r.value, ast.Constant):
            vals.add(r.value.value)
        else:
            return False, vals
    return bool(rs), vals


def _literal_index(sl: ast.AST) -> bool:
    """True iff this subscript index is a LITERAL row/col/slice.

    `grid[63, :]` yes -- a fixed band. `grid[grid != 0]` NO: that is a whole-board boolean mask,
    a global test wearing a subscript's clothes, and counting it as a fixed region would file
    every "all non-background cells are one colour" predicate under the wrong cluster.
    `grid[r, c]` no -- the row/col came from somewhere.
    """
    for n in ast.walk(sl):
        if isinstance(n, (ast.Compare, ast.BoolOp, ast.Call, ast.Name)):
            return False
    return True


def _has_text(fn: ast.FunctionDef, *frags: str) -> bool:
    s = ast.unparse(fn)
    return any(f in s for f in frags)


CLUSTER_DOC = {
    "A_DECLINED": "unconditional `return False` -- no state can satisfy it",
    "D_NO_PREDICATE": "generation defect: no return at all, or raises NameError when called",
    "I_CONSTANT_TRUE": "unconditional `return True`",
    "G_CONNECTIVITY": "flood-fill / 'every colour forms one connected region'",
    "C_UNIFORMITY": "'the whole board becomes one colour'",
    "E_FIXED_BAND": "decides on a LITERAL grid region (a HUD row, an edge column, a rectangle)",
    "B_COLOUR_ELIMINATION": "'colour X is gone' / 'X occurs N times' over the whole board",
    "F_OBJECT_POSITION": "locate an object by colour, then test its coordinates",
    "H_OTHER": "unclassified",
}


def classify(fn: ast.FunctionDef) -> tuple[str, str]:
    """-> (cluster, note). First match wins; the order below is the whole design.

    Generation defects are tested first because a predicate that cannot run has no semantics to
    cluster. Constant returns come next for the same reason. Uniformity is tested BEFORE fixed
    band because `np.all(grid == grid[0, 0])` contains a literal subscript but is a global claim.
    """
    free = _free_names(fn)
    all_const, vals = _const_returns(fn)
    returns = [n for n in ast.walk(fn) if isinstance(n, ast.Return)]

    if not returns:
        return "D_NO_PREDICATE", "no return statement at all (falls off the end -> None)"
    if free:
        return "D_NO_PREDICATE", f"raises NameError on undefined name(s): {sorted(free)}"
    if all_const and vals == {False}:
        doc = (ast.get_docstring(fn) or "").lower()
        said = any(
            k in doc
            for k in ("no win state", "win state provided", "no one has won", "not provided")
        )
        return "A_DECLINED", (
            "model states in its own docstring that no win state was provided"
            if said
            else "unconditional `return False`, no reason given"
        )
    if all_const and vals == {True}:
        return "I_CONSTANT_TRUE", "unconditionally true"
    if _has_text(fn, "visited", "queue", "stack") and _has_text(
        fn, "(0, 1)", "(1, 0)", "dr, dc", "dx, dy"
    ):
        return "G_CONNECTIVITY", "flood-fill / connected-component structural check"
    if _has_text(
        fn,
        "np.all(grid ==",
        "np.all(non_zero ==",
        "len(unique_colors) == 1",
        "len(non_zero_colors) == 1",
        "np.all(non_zero == non_zero[0])",
    ):
        return "C_UNIFORMITY", "'the whole board becomes one colour'"

    literal_subs, var_subs = [], []
    for n in ast.walk(fn):
        if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id == "grid":
            (literal_subs if _literal_index(n.slice) else var_subs).append(ast.unparse(n))
    if literal_subs and not var_subs:
        return "E_FIXED_BAND", f"literal grid region(s): {sorted(set(literal_subs))[:4]}"
    if _has_text(fn, "not np.any(grid ==", "np.sum(grid ==", "np.any(grid ==", "np.count_nonzero"):
        return "B_COLOUR_ELIMINATION", "'colour X is gone / occurs N times' over the whole board"
    if _has_text(fn, "np.argwhere", "np.where") or (
        var_subs and _has_text(fn, "for r in range", "for y in range")
    ):
        return "F_OBJECT_POSITION", "locate an object, then test its coordinates"
    return "H_OTHER", "unclassified"


def build_records() -> list[dict]:
    rows = json.loads(METRIC_VALIDITY.read_text())["rows"]
    out: list[dict] = []
    for r in rows:
        src, where = source_for(r)
        rec = {
            "cell": r["cell"],
            "game": r["game"],
            "corpus": r["corpus"],
            "goal_kind": r["goal_kind"],
            "goal_satisfiable": r["goal_satisfiable"],
            "plan_found": r["plan_found"],
            # "live" == the engine changes SOMETHING at the root. An inert engine's goal is moot:
            # nothing is reachable, so no predicate could have been satisfied.
            "live": bool(r.get("engine_changes_anything_at_root")),
            "path": where,
        }
        if src is None:
            rec["error"] = where
            out.append(rec)
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError as exc:
            rec["error"] = f"SyntaxError:{exc.msg}"
            out.append(rec)
            continue
        defs = [
            n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "is_level_complete"
        ]
        rec["n_defs"] = len(defs)
        # The `_combine_world_model` fingerprint: engine block + goal block, each carrying its
        # own numpy import.
        rec["split_induce_signature"] = src.startswith("import numpy as np\n\nimport numpy as np")
        if not defs:
            rec["error"] = "no_is_level_complete_def"
            out.append(rec)
            continue
        chosen = defs[-1]
        rec["source"] = ast.get_source_segment(src, chosen) or ast.unparse(chosen)
        rec["normalized"] = ast.unparse(chosen)
        rec["source_sha256_16"] = hashlib.sha256(rec["source"].encode()).hexdigest()[:16]
        if len(defs) > 1:
            rec["shadowed_definitions"] = [ast.unparse(d) for d in defs[:-1]]
        cluster, note = classify(chosen)
        rec["cluster"], rec["cluster_note"] = cluster, note
        # Provable without running anything: these two clusters can never return True, so the
        # planner's search was hopeless before it expanded a single node.
        rec["never_true_by_construction"] = cluster in ("A_DECLINED", "D_NO_PREDICATE")
        out.append(rec)
    return out

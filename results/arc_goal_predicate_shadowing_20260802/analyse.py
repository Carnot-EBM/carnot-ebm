"""Step 1: find every engine that defines `is_level_complete` more than once, and split it.

THE CLAIM UNDER TEST. On the split-induce fallback path
(`arc_executable_world_model.LocalGGUFProposer.induce`, the branch taken when the
combined engine+goal call fails), two generations are CONCATENATED by
`_combine_world_model`: an engine-only block, then a goal-only block. The engine-only
prompt is `induce_prompt(...)` -- it carries the observed transitions and the level's
opening grid. The goal-only prompt is `_goal_only_prompt(...)`, which by default
(`_goal_prompt_transitions_on()` is OFF) carries NO transitions. If the model writes an
`is_level_complete` in the engine-only response as well -- and it does, because the
combined interface is described in the base prompt -- the concatenation puts the
goal-only version SECOND, and Python binds the last top-level definition. The
evidence-carrying predicate is therefore shadowed by the evidence-free one.

WHAT THIS FILE DOES, AND DOES NOT DO. It only PARSES and CLASSIFIES: no code is
executed here, because a generated `is_level_complete` may loop forever or raise, and
importing it into this interpreter is exactly the hazard `measure_worker.py` exists to
contain. The binding rule below is not re-derived either -- it defers to the SHIPPED
resolver, `arc_engine_static_validation._find_function`, so this analysis cannot
disagree with the validator the project already ships about which definition runs.

THE `evidence_grounded` LABEL IS A HEURISTIC AND IS NOT THE RESULT. It is reported so
the artifact can say which cells the mechanism is even capable of helping, but the
verdict comes from `measure_worker.py` running both predicates through the shipped
gate. A syntactic label that agrees with a dynamic measurement is corroboration; a
syntactic label on its own would be the assertion this task was told not to make.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "python"))

CORPORA = {
    "ab_change_fidelity": REPO
    / "results"
    / "arc_object_perception_ab_change_fidelity_20260801"
    / "engines",
    "induce_bestofn": REPO / "results" / "arc_induce_bestofn_20260731" / "harness" / "bon",
}

GOAL_FN = "is_level_complete"
ENGINE_FN = "engine"

# A predicate whose body mentions a concrete board coordinate, a concrete colour compared
# against a concrete row/column slice, or a named object/goal colour pair, is REFERRING TO
# THE BOARD IT WAS SHOWN. A predicate phrased purely as "all cells equal", "all non-zero
# cells equal" or a connectivity sweep is a generic ARC trope that needs no evidence at all.
_GROUNDED_PAT = re.compile(
    r"grid\s*\[\s*\d+|"  # an explicit row/cell index, e.g. grid[63]
    r"grid\s*\[\s*[^]]*\d+\s*:\s*\d+|"  # an explicit slice, e.g. grid[:, 62:64]
    r"np\.argwhere\s*\(\s*grid\s*==\s*\d+\s*\)|"  # locating a specific colour's cells
    r"np\.sum\s*\(\s*grid\s*\[",  # counting within a specific band
)
_TROPE_PAT = re.compile(
    r"np\.all\s*\(\s*grid\s*==\s*grid\s*\[|"  # all cells equal grid[0,0]
    r"non_zero\s*==\s*non_zero\s*\[\s*0|"  # all non-zero cells one colour
    r"len\s*\(\s*np\.unique|"  # unique-colour count
    r"unique_colors",  # the connectivity/uniformity sweep vocabulary
)


def _is_constant_false(fn: ast.FunctionDef) -> bool:
    """Does this predicate do nothing but `return False` (modulo docstring/comments)?"""
    body = [
        s for s in fn.body if not (isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant))
    ]
    return (
        len(body) == 1
        and isinstance(body[0], ast.Return)
        and isinstance(body[0].value, ast.Constant)
        and body[0].value.value is False
    )


def _classify(src_segment: str, fn: ast.FunctionDef) -> str:
    if _is_constant_false(fn):
        return "A_DECLINED"
    if _GROUNDED_PAT.search(src_segment):
        return "GROUNDED"
    if _TROPE_PAT.search(src_segment):
        return "TROPE"
    return "OTHER"


def _static_defects(code: str, fn_name: str) -> list[dict[str, Any]]:
    """The SHIPPED static validator's verdict on this function, not a re-implementation."""
    from carnot.agentic import arc_engine_static_validation as sv

    out = []
    for d in sv.missing_return_defects(code, fn_name):
        out.append({"kind": d.kind, "detail": d.detail, "line": d.line})
    return out


def _shipped_binding_index(code: str, defs: list[ast.FunctionDef]) -> int | None:
    """Which definition does the project's own resolver say the caller gets?"""
    from carnot.agentic import arc_engine_static_validation as sv

    bound = sv._find_function(ast.parse(code), GOAL_FN)
    if bound is None:
        return None
    for i, d in enumerate(defs):
        if d.lineno == bound.lineno:
            return i
    return None


def _iter_engine_files() -> list[tuple[str, str, Path]]:
    """Every candidate world model in both corpora.

    The two corpora are deliberately DIFFERENT SHAPES, and that difference is the
    control this analysis rests on:

    * `ab_change_fidelity` holds `world_model.py` files as they were WRITTEN TO DISK by
      `_write_world_model`, i.e. AFTER `_combine_world_model` concatenated the two
      halves on the split path. These are the files that can exhibit shadowing.
    * `induce_bestofn` holds RAW single completions (`gpu1/<game>_k<n>.txt`) from the
      COMBINED call -- one prompt, one response, no concatenation anywhere. If
      shadowing were the model spontaneously redefining the function, it would show up
      here too. If it is the concatenation, this corpus is clean. Including it is what
      makes the mechanism claim falsifiable rather than merely consistent.
    """
    found = []
    for corpus, root in CORPORA.items():
        if not root.exists():
            continue
        for path in sorted(root.rglob("world_model.py")):
            rel = path.relative_to(root)
            cell = rel.parts[0] if len(rel.parts) > 1 else path.stem
            found.append((corpus, cell, path))
        for path in sorted(root.rglob("*_k*.txt")):
            found.append((corpus, path.stem, path))
    return found


def main() -> int:
    rows: list[dict[str, Any]] = []
    games: set[str] = set()
    totals = {"files": 0, "parse_error": 0, "single": 0, "multi": 0}

    for corpus, cell, path in _iter_engine_files():
        totals["files"] += 1
        code = path.read_text()
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            totals["parse_error"] += 1
            rows.append(
                {
                    "corpus": corpus,
                    "cell": cell,
                    "path": str(path.relative_to(REPO)),
                    "parse_error": f"{exc.msg} (line {exc.lineno})",
                    "n_goal_defs": None,
                }
            )
            continue

        defs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == GOAL_FN]
        np_imports = [
            n
            for n in tree.body
            if isinstance(n, ast.Import) and any(a.name == "numpy" for a in n.names)
        ]
        # AB cells are `<game>__r<n>__<arm>`; best-of-n candidates are `<game>_k<n>`.
        game = cell.split("__")[0].split("_k")[0]
        # Only the AB corpus contributes root grids to capture: the best-of-n corpus is a
        # parse-only control and never reaches the dynamic gate.
        if corpus == "ab_change_fidelity":
            games.add(game)

        row: dict[str, Any] = {
            "corpus": corpus,
            "cell": cell,
            "game": game,
            "path": str(path.relative_to(REPO)),
            "n_goal_defs": len(defs),
            "n_numpy_imports": len(np_imports),
            "split_induce_signature": len(np_imports) > 1,
            "engine_defect": _static_defects(code, ENGINE_FN),
        }
        if len(defs) < 2:
            totals["single"] += 1
            rows.append(row)
            continue

        totals["multi"] += 1
        bound_ix = _shipped_binding_index(code, defs)
        row["bound_index"] = bound_ix
        row["bound_is_last"] = bound_ix == len(defs) - 1
        row["defs"] = []
        for i, fn in enumerate(defs):
            seg = ast.get_source_segment(code, fn) or ""
            row["defs"].append(
                {
                    "index": i,
                    "lineno": fn.lineno,
                    "role": "bound" if i == bound_ix else "shadowed",
                    "n_lines": len(seg.splitlines()),
                    "sha256_16": hashlib.sha256(seg.encode()).hexdigest()[:16],
                    "classification": _classify(seg, fn),
                    "constant_false": _is_constant_false(fn),
                    "static_defects": _static_defects(seg, GOAL_FN),
                    "source": seg,
                }
            )
        rows.append(row)

    (HERE / "corpus_games.json").write_text(json.dumps(sorted(games), indent=2) + "\n")
    (HERE / "analysis.json").write_text(
        json.dumps({"totals": totals, "rows": rows}, indent=2) + "\n"
    )
    multi = [r for r in rows if (r.get("n_goal_defs") or 0) > 1]
    print(json.dumps(totals))
    print(f"multi-definition cells: {len(multi)}")
    print(f"games: {len(games)}")
    same = sum(
        1 for r in rows if r.get("split_induce_signature") == ((r.get("n_goal_defs") or 0) > 1)
    )
    print(f"split-induce signature agrees with multi-def on {same}/{totals['files']} files")
    for corpus in CORPORA:
        sub = [r for r in rows if r["corpus"] == corpus and r.get("n_goal_defs") is not None]
        m = sum(1 for r in sub if r["n_goal_defs"] > 1)
        print(f"  {corpus}: {m}/{len(sub)} multi-definition")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Score every generated engine BEHAVIOURALLY, because "returns on all paths" is gameable.

WHY THIS EXISTS (found by reading the output, not by the metric). The refactor lane's first
completion scored `usable_engine = True` under the AST check -- accepted by `generate()`, parses,
and `engine()` returns on every path. Reading it shows what it actually is:

    def engine(grid, action, data):
        x = data.get('x', 0); y = data.get('y', 0)
        rows = len(grid); cols = len(grid[0]) if rows > 0 else 0
        if action == 6:
            return grid
        return grid

An IDENTITY engine. It returns the grid unmodified on every path, so it satisfies the
return-on-all-paths check trivially while modelling nothing. Reporting that as a Phase-1 win
would be exactly the over-claim the repo's own `change_fidelity` docstring warns about
("identity engines score 0.0 here by construction"), arrived at by trusting a metric instead of
reading what it scored.

So every engine is now RUN against ft09's real captured transitions:

  * `engine_changes_anything` -- does it ever produce an output different from its input? An
    engine that cannot is degenerate no matter how cleanly it returns.
  * `heldout_exact` / `n_exact` -- transitions reproduced byte-exactly (the live gate's metric).
  * `cell_recall` -- of the cells reality changed, the share the engine got right.
  * `engine_raised` -- exceptions are a datum, not a zero.

Pure Python + numpy over the captured transitions; no GPU, no server.
"""

from __future__ import annotations

import json
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "/home/ianblenke/github.com/ianblenke/carnot"
sys.path.insert(0, os.path.join(REPO, "python"))

TRANS_PKL = os.path.join(HERE, "ft09_transitions.pkl")


def load_transitions() -> list:
    """The 25 transitions ft09's induction point was built from, captured from the agent itself."""
    with open(TRANS_PKL, "rb") as fh:
        return pickle.load(fh)


def _shown_transition_indices(trans: list) -> list[int]:
    """Which transitions the induce prompt actually displays.

    Determined by matching each transition's EXACT rendered `data={'x': N, 'y': M}` string
    against the captured prompt, rather than by re-deriving `_transitions_block`'s selection
    rule -- reading the prompt that was really sent cannot drift from it. Measured for ft09:
    9 of 25 shown, 16 held out.
    """
    try:
        prompt = open(os.path.join(HERE, "prompts", "prompt2_engine.txt")).read()
    except OSError:
        return []
    out = []
    for i, t in enumerate(trans):
        d = t.data or {}
        if f"data={{'x': {d.get('x')}, 'y': {d.get('y')}}}" in prompt:
            out.append(i)
    return out


def score_code(code: str, trans: list) -> dict:
    from carnot.agentic import arc_executable_world_model as e3  # noqa: F401  (numpy env parity)

    out = {
        "engine_raised": None,
        "engine_changes_anything": False,
        "n_transitions_scored": 0,
        "n_exact": 0,
        "heldout_exact": 0.0,
        "cell_recall": 0.0,
        "n_true_changed_cells": 0,
        "n_correct_changed_cells": 0,
    }
    ns: dict = {"np": np, "numpy": np}
    try:
        exec(compile(code, "<engine>", "exec"), ns)  # noqa: S102 -- scoring generated code is the job
    except Exception as exc:
        out["engine_raised"] = f"import/exec: {type(exc).__name__}: {exc}"[:200]
        return out
    fn = ns.get("engine")
    if not callable(fn):
        out["engine_raised"] = "no callable `engine` after exec"
        return out

    # HELD-OUT SPLIT. The induce prompt SHOWS 9 of the 25 transitions (identified by their exact
    # rendered `data={'x': N, 'y': M}` string appearing in the prompt), so a cell-recall over all
    # 25 is partly in-sample. Quoting it as if it were held-out would overstate any positive
    # result, which is the specific over-claim risk in this sweep now that one arm has a positive.
    shown_idx = set(_shown_transition_indices(trans))
    n_true = n_ok = 0
    ho_true = ho_ok = 0
    for _i, t in enumerate(trans):
        before = np.asarray(t.grid)
        after = np.asarray(t.next_grid) if getattr(t, "next_grid", None) is not None else None
        if after is None:
            continue
        try:
            pred = fn(before.copy(), int(t.action), dict(t.data or {}))
        except Exception as exc:
            out["engine_raised"] = out["engine_raised"] or f"{type(exc).__name__}: {exc}"[:200]
            continue
        pred = np.asarray(pred)
        out["n_transitions_scored"] += 1
        if pred.shape == before.shape and not np.array_equal(pred, before):
            out["engine_changes_anything"] = True
        if pred.shape == after.shape and np.array_equal(pred, after):
            out["n_exact"] += 1
        if pred.shape == after.shape == before.shape:
            changed = after != before
            n_true += int(changed.sum())
            n_ok += int((changed & (pred == after)).sum())
            if _i not in shown_idx:
                ho_true += int(changed.sum())
                ho_ok += int((changed & (pred == after)).sum())
    out["n_true_changed_cells"] = n_true
    out["n_correct_changed_cells"] = n_ok
    out["cell_recall"] = round(n_ok / n_true, 4) if n_true else 0.0
    out["n_transitions_shown_in_prompt"] = len(shown_idx)
    out["heldout_true_changed_cells"] = ho_true
    out["heldout_correct_changed_cells"] = ho_ok
    out["heldout_cell_recall"] = round(ho_ok / ho_true, 4) if ho_true else 0.0
    out["heldout_exact"] = (
        round(out["n_exact"] / out["n_transitions_scored"], 4)
        if out["n_transitions_scored"] else 0.0
    )
    return out


def main() -> int:
    from carnot.agentic import arc_executable_world_model as e3

    trans = load_transitions()
    print(f"scoring against {len(trans)} captured ft09 transitions")
    dirs = sys.argv[1:] or ["sweep", "sweep_combined", "sweep_refactor", "sweep_sampler"]
    rows = []
    for d in dirs:
        p = os.path.join(HERE, d, "sweep.json")
        if not os.path.exists(p):
            continue
        doc = json.load(open(p))
        for r in doc.get("rows") or []:
            fn = r.get("completion_file")
            if not fn:
                continue
            fp = os.path.join(HERE, d, fn)
            if not os.path.exists(fp):
                continue
            text = open(fp).read()
            code = e3._extract_python(text) or (
                text.strip() if d != "sweep_refactor" else ""
            )
            st = score_code(code, trans) if code else {"engine_raised": "no code extracted"}
            rows.append({
                "lane": d, "arm": r.get("arm", "shipped"), "prompt": r["prompt"],
                "budget": r["budget"], "attempt": r["attempt"],
                "ast_returns_on_all_paths": r.get("engine_returns_on_all_paths"),
                "generate_would_accept": r.get("generate_would_accept"),
                "completion_file": fn, **st,
            })

    hdr = (f"{'lane':>16}{'arm':>18}{'bud':>7}{'a':>3}{'accept':>7}{'returns':>8}"
           f"{'changes':>8}{'exact':>7}{'cellrec':>8}  raised")
    print(hdr); print("-" * len(hdr))
    for r in sorted(rows, key=lambda r: (r["lane"], r["arm"], r["budget"], r["attempt"])):
        print(f"{r['lane']:>16}{r['arm']:>18}{r['budget']:>7}{r['attempt']:>3}"
              f"{str(r.get('generate_would_accept')):>7}{str(r.get('ast_returns_on_all_paths')):>8}"
              f"{str(r.get('engine_changes_anything')):>8}{str(r.get('n_exact')):>7}"
              f"{str(r.get('cell_recall')):>8}  {str(r.get('engine_raised'))[:60]}")

    with open(os.path.join(HERE, "engine_scores.json"), "w") as fh:
        json.dump(rows, fh, indent=2, sort_keys=True)
    n_real = sum(1 for r in rows
                 if r.get("generate_would_accept") and r.get("ast_returns_on_all_paths")
                 and r.get("engine_changes_anything"))
    print(f"\nNON-DEGENERATE usable engines (accepted AND returns AND changes something): {n_real}"
          f" of {len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

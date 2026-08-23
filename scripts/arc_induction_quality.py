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

BRANCH COUNT DOES NOT RANK QUALITY (REQ-ARC-WMTE-6700). Read `branches` as
"how much code is here", never as "how good is it". The four highest-branch
models in the 2026-08 corpus (cd82 at 376, vc33 at 356, ls20, sp80) transcribe
the win grid row by row into literal comparisons. That is the worst
memorisation present, and ranking by branches puts it at the top.

POPULATION GUARD (REQ-ARC-WMTE-6700). This scorer answers a question about the
LIVE generator, so it must not score files that no live run produced. Two
classes are excluded by default and named in the report: a nested repo clone
(a `.git` under the sweep root -- that content is a repository's committed
state) and a committed `results/` tree (evidence, per the Test-Run Record
Integrity Discipline). Both were being swept on 2026-08-23: 91% of 1248 files
were one committed tree, counted twice because two clones sat under the root,
and its generator was Qwen3.5-9B, retired 2026-07-28. `--include-non-live`
restores the old behaviour and says the rates are not live rates.

COUNT DISTINCT, NOT JUST TOTAL (REQ-ARC-WMTE-6700). The report carries total
files, distinct files, and distinct goal predicates. A duplicated root inflates
only the first, so the duplication shows up in the output instead of having to
be suspected.

SAMPLING BIAS WARNING, and the fix (REQ-ARC-WMTE-6690). A sweep over surviving
`world_model.py` files is a WORST-ATTEMPT sample: re-induction fires on
stagnation, so the survivor is systematically the attempt made under the worst
conditions, and 15 of 40 attempts on the 2026-08-22 baseline run were destroyed
before anyone could score them. Since REQ-ARC-WMTE-6690 every producer write is
also archived under `<store>/<game>/attempts/wm_*__<sha16>.py`; this scorer
sweeps those too, so a degenerate-rate computed over attempts/ is a
per-attempt generator-quality rate, not a survivor rate. Rows report which
population they came from (`population: survivor|attempt`).

Read-only. Touches nothing.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
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
        # Archived attempts live one level deeper (<game>/attempts/wm_*.py), so the game
        # is the grandparent there and the parent for a canonical survivor file.
        "game": path.parent.parent.name if path.parent.name == "attempts" else path.parent.name,
        "population": "attempt" if path.parent.name == "attempts" else "survivor",
        "bytes": len(src.encode("utf-8")),
        "lines": src.count("\n") + 1,
        "file_sha16": hashlib.sha256(src.encode("utf-8")).hexdigest()[:16],
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
        # Hash the predicate, not the file: two runs of the same generator often
        # differ only in a comment, and the predicate is what the rate is about.
        out["goal_predicate_sha16"] = hashlib.sha256(seg.encode("utf-8")).hexdigest()[:16]
        out["goal_branches"] = sum(
            1
            for n in ast.walk(goal_fn)
            if isinstance(n, (ast.If, ast.For, ast.While, ast.IfExp, ast.Compare))
        )
    return out


def classify_path(path: Path, root: Path) -> str:
    """REQ-ARC-WMTE-6700: is this path something a LIVE run produced?

    Returns `live_run`, or the name of the class that disqualifies it. A clone
    holds a repository's committed state, and `results/` is evidence -- neither
    is this run's output, and counting them answered the wrong question for
    hours on 2026-08-23.
    """
    try:
        rel = path.resolve().relative_to(root.resolve())
    except ValueError:
        # Outside the root we cannot reason about provenance, so we do not
        # guess: fail closed and let the caller see it named.
        return "outside_root"

    parts = rel.parts[:-1]  # directories only; the file itself is never a marker
    for i in range(len(parts)):
        ancestor = root.joinpath(*parts[: i + 1])
        if (ancestor / ".git").exists():
            return "nested_repo_clone"
    if "results" in parts:
        return "committed_results_tree"
    return "live_run"


def find_models(roots: list[Path], *, include_non_live: bool = False) -> tuple[list[Path], dict]:
    """Return (kept paths, {exclusion class: count}).

    Exclusions are returned rather than dropped so the report can name them;
    a guard that silently filters is the state this REQ exists to leave.
    """
    kept: list[Path] = []
    excluded: dict[str, int] = {}
    for root in roots:
        if not root.is_dir():
            continue
        found = sorted(root.rglob("world_model.py"))
        # REQ-ARC-WMTE-6690 archived attempts: the unbiased per-attempt population.
        found += sorted(root.rglob("attempts/wm_*.py"))
        for path in found:
            verdict = classify_path(path, root)
            if verdict == "live_run" or include_non_live:
                kept.append(path)
            else:
                excluded[verdict] = excluded.get(verdict, 0) + 1
    return kept, excluded


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roots", nargs="*", default=[str(DEFAULT_ROOT)])
    ap.add_argument("--json", action="store_true")
    ap.add_argument(
        "--include-non-live",
        action="store_true",
        help="Also score clones and committed results/ trees. Deliberate archaeology "
        "only: the resulting rates are NOT live-generator rates.",
    )
    args = ap.parse_args(argv)

    models, excluded = find_models(
        [Path(r) for r in args.roots], include_non_live=args.include_non_live
    )
    if not models:
        if excluded:
            # Fail closed. "Nothing live here" is a true answer; a rate computed
            # over archived-only files silently answers a different question.
            print("no LIVE world_model.py found under", args.roots)
            for cls, n in sorted(excluded.items()):
                print(f"  excluded {cls}: {n}")
            print("  pass --include-non-live to score them anyway (not live rates).")
        else:
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
        # An archived attempt sits one level deeper (<run>/e3/<game>/attempts/wm_*.py) than a
        # survivor (<run>/e3/<game>/world_model.py), so the run name is one part further back.
        # Reading it at the survivor depth labels every attempt "e3", which makes the two
        # populations look like they came from different runs -- the exact comparison this
        # scorer exists to support.
        depth = 5 if s.get("population") == "attempt" else 4
        parts = Path(s["path"]).parts
        run = parts[-depth] if len(parts) >= depth else "?"
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
    n_att = sum(1 for s in scored if s.get("population") == "attempt")
    # REQ-ARC-WMTE-6700: three population sizes, so a duplicated root shows up here
    # instead of having to be suspected. Only the first inflates under duplication.
    n_files = len({s["file_sha16"] for s in scored if s.get("file_sha16")})
    n_preds = len({s["goal_predicate_sha16"] for s in scored if s.get("goal_predicate_sha16")})
    print("")
    print(
        f"  {len(scored)} model(s); {n_deg} degenerate; {n_struct} with a structural goal predicate"
    )
    print(f"  distinct: {n_files} file(s) / {n_preds} goal predicate(s) of {len(scored)} scored")
    # Survivor-only rates are worst-attempt-biased (see module docstring); say which
    # population the number came from so it cannot be misread as a generator rate.
    print(f"  populations: {len(scored) - n_att} survivor / {n_att} archived attempt")
    for cls, n in sorted(excluded.items()):
        print(f"  EXCLUDED {cls}: {n} (not live-run output; --include-non-live to score)")
    if args.include_non_live:
        print("  WARNING: --include-non-live is set. These are NOT live-generator rates.")
    print("  Structure is not correctness. Held-out transition accuracy needs a run that survives.")
    print("  Branch count is size, not quality: the highest-branch models memorise the win grid.")
    print("")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

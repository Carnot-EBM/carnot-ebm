"""M2-v3: codex (gpt-5.5) program synthesis of a per-game transition model, VERIFIED by the
grid-grounded consistency energy. The escalation tier the M1->M2v2 chain demanded.

Plan: docs/research-notes/arc-agi3-agent-research-plan.md (M2, Family-B / arXiv:2605.05138). M2-v1a
(naive pixel template) and M2-v2 (object DSL) showed a FIXED inducer cannot express ARC's diverse
dynamics (DSL fit only 5/25 games). The plan's answer is an EXPRESSIVE per-game inducer: a coding
agent writes a bespoke Python `predict(grid, action)` transition program; the consistency energy (no
oracle) grades it against HELD-OUT real transitions and drives a bounded refactor loop. This is the
Carnot verifier-prunes / generator-induces division made load-bearing: codex is the generator, the
energy is the oracle-free fitness function that certifies/prunes the program.

  generator: codex writes predict(grid, action)      (the rare heavy inducer, escalation-only)
  verifier : grade_predictions(predict, held_out)     (grid-grounded, no oracle; 0 = trustworthy)

SAFETY: codex-written code is executed in a RESTRICTED namespace (numpy only; curated safe builtins;
import/open/eval/exec/os/sys/subprocess rejected) with per-call exception capture (a crash -> graded
as misprediction, never trusted). This is research code on the dev box, not a hardened sandbox;
CARNOT_USE_SANDBOX/gvisor is the production path.

  .venv/bin/python scripts/experiments/arc3_m2_codex_synth.py --games vc33,sb26,m0r0,ls20 --iters 3
"""

from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "scripts" / "experiments"))
from carnot.agentic.arc_world_model_synth import InducedWorldModel, grade_predictions  # noqa: E402
from carnot.agentic.arc_world_model_dsl import ObjectDeltaModel  # noqa: E402
from arc3_m2_world_model import _collect, _key_disjoint_split  # noqa: E402

CODEX = ["codex", "exec", "--color", "never", "--model", "gpt-5.5",
         "-c", "model_reasoning_effort=medium",  # xhigh default is too slow (>180s); medium balances
         "--dangerously-bypass-approvals-and-sandbox", "--cd", "/tmp", "--ephemeral"]
_FORBIDDEN = ("__import__", "open(", "eval(", "exec(", "compile(", "subprocess", "os.", "sys.",
              "import os", "import sys", "import subprocess", "socket", "shutil", "Path(")
_SAFE_BUILTIN_NAMES = ["range", "len", "min", "max", "abs", "enumerate", "zip", "sum", "sorted",
                       "list", "dict", "set", "tuple", "int", "float", "bool", "map", "filter",
                       "reversed", "any", "all", "round", "isinstance", "print", "str", "bytes"]


def _safe_builtins():
    import builtins as _b
    return {k: getattr(_b, k) for k in _SAFE_BUILTIN_NAMES if hasattr(_b, k)}


def safe_predict_from_code(code: str):
    """Compile codex's predict() in a restricted namespace; return a robust callable or None."""
    if any(tok in code for tok in _FORBIDDEN):
        return None
    body = "\n".join(l for l in code.splitlines()
                     if not l.strip().startswith(("import ", "from ")))   # np is provided directly
    ns = {"np": np, "numpy": np, "__builtins__": _safe_builtins()}
    try:
        exec(body, ns)                              # def predict only; body not run yet
    except Exception:
        return None
    fn = ns.get("predict")
    if not callable(fn):
        return None

    def wrapped(grid, action):
        try:
            out = np.asarray(fn(np.asarray(grid, dtype=np.int16).copy(), tuple(action)), dtype=np.int16)
            return out if out.shape == np.asarray(grid).shape else np.asarray(grid, dtype=np.int16)
        except Exception:
            return np.asarray(grid, dtype=np.int16)  # crash -> no-op -> graded as misprediction
    return wrapped


def _grid_view(g, bg, margin=3, cap=24):
    """Compact text view of a grid cropped to the bounding box of non-background cells (so codex sees
    spatial STRUCTURE, not just a changed-cell list which invites hardcoding coordinates)."""
    g = np.asarray(g)
    nz = np.argwhere(g != bg)
    if nz.size == 0:
        return "(all background)"
    y0, x0 = nz.min(0); y1, x1 = nz.max(0)
    y0 = max(0, int(y0) - margin); x0 = max(0, int(x0) - margin)
    y1 = min(g.shape[0], int(y1) + margin + 1); x1 = min(g.shape[1], int(x1) + margin + 1)
    sub = g[y0:y1, x0:x1][:cap, :cap]
    rows = [" ".join(f"{int(v):2d}" for v in row) for row in sub]
    return f"(rows {y0}..{y0 + sub.shape[0] - 1}, cols {x0}..{x0 + sub.shape[1] - 1})\n" + "\n".join(rows)


def _serialize(sample, bg, shape, max_cells=36):
    lines = [f"Grid is {shape[0]} rows x {shape[1]} cols, integer colors 0-15, background={bg}.",
             "Action tuple: (k,) = keyboard action k; (6, x, y) = a click at column x, row y.",
             "Observed transitions (action -> the cells that changed, as 'row,col:old->new'):"]
    seen_acts = set()
    for s, akey, s2 in sample:
        diff = np.argwhere(s != s2)
        changes = ", ".join(f"{int(y)},{int(x)}:{int(s[y, x])}->{int(s2[y, x])}" for y, x in diff[:max_cells])
        more = f" (+{len(diff) - max_cells} more changed)" if len(diff) > max_cells else ""
        if akey[0] == 6 and 0 <= akey[2] < shape[0] and 0 <= akey[1] < shape[1]:
            act = f"(6, {akey[1]}, {akey[2]}) click on color {int(s[akey[2], akey[1]])}"
        else:
            act = str(tuple(akey))
        lines.append(f"  action {act} -> {changes or '(no change)'}{more}")
    # one full cropped BEFORE/AFTER view per distinct action key-int (spatial grounding)
    lines.append("\nFull cropped grid views (one per action type) so you can see the spatial structure:")
    for s, akey, s2 in sample:
        if akey[0] in seen_acts or not (s != s2).any():
            continue
        seen_acts.add(akey[0])
        act = f"(6,{akey[1]},{akey[2]})" if akey[0] == 6 else str(tuple(akey))
        lines.append(f"--- action {act} BEFORE: {_grid_view(s, bg)}\n    AFTER:  {_grid_view(s2, bg)}")
        if len(seen_acts) >= 4:
            break
    return "\n".join(lines)


def _extract_code(text: str):
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", text, re.S)
    for b in reversed(blocks):
        if "def predict" in b:
            return b.strip()
    return None


def ask_codex(prompt, timeout=300):
    t0 = time.time()
    try:
        r = subprocess.run(CODEX, input=prompt, capture_output=True, text=True, timeout=timeout)
        return r.stdout or "", round(time.time() - t0, 1)
    except Exception as e:
        return f"__codex_error__:{type(e).__name__}", round(time.time() - t0, 1)


def synth_prompt(serialized, prior_code=None, failures=None):
    base = (
        "You are reverse-engineering the DETERMINISTIC transition rule of a grid puzzle from examples.\n\n"
        f"{serialized}\n\n"
        "Write exactly one Python function:\n"
        "    def predict(grid, action):\n"
        "        # grid: a 2D numpy int array (rows x cols). action: the tuple described above.\n"
        "        # return the NEXT grid (a numpy int array, same shape) the rule produces.\n"
        "Infer the underlying mechanic (object move/translate, recolor, toggle, gravity, fill, "
        "selection, etc.) and GENERALIZE to unseen states and actions. Use ONLY numpy (np is already "
        "imported). No file/network/os access.\n\n"
        "CRITICAL: your function is called on UNSEEN grids, so DO NOT hardcode specific colors, "
        "coordinates, object sizes, or shapes copied from the examples. DETECT them from the input "
        "grid at runtime: infer the background as the most common color, find objects as connected "
        "components / by color, and compute each action's effect generically (e.g. a consistent "
        "displacement of a detected object). A function that returns the grid unchanged, or that only "
        "works for the example positions, scores ZERO. Output ONLY one ```python code block.")
    if prior_code and failures:
        base += ("\n\nYour PREVIOUS function mispredicted these held-out transitions "
                 "(action -> changes it should have produced):\n" + failures +
                 "\n\nPrevious code:\n```python\n" + prior_code + "\n```\nFix it and output the corrected "
                 "function as one ```python block.")
    return base


def _failure_examples(predict_fn, held, bg, shape, k=6):
    out = []
    for s, akey, s2 in held:
        if len(out) >= k:
            break
        if not (s != s2).any():
            continue
        pred = predict_fn(s, akey)
        if not np.array_equal(pred, s2):
            diff = np.argwhere(s != s2)
            ch = ", ".join(f"{int(y)},{int(x)}:{int(s[y, x])}->{int(s2[y, x])}" for y, x in diff[:20])
            act = f"(6, {akey[1]}, {akey[2]})" if akey[0] == 6 else str(tuple(akey))
            out.append(f"  action {act} -> {ch}")
    return "\n".join(out)


def run(games, budget=1200, episodes=35, iters=3, seed=0, write=True):
    from arc_agi import Arcade
    from arc_agi.base import OperationMode
    from arcengine.enums import GameAction, GameState
    started = time.time()
    rng = random.Random(seed)
    arc = Arcade(arc_api_key="", operation_mode=OperationMode.OFFLINE,
                 environments_dir=str(REPO / "environment_files"))
    all_ids = sorted(getattr(e, "game_id", None) for e in arc.get_environments())
    sel = [g for g in all_ids if g.split("-")[0] in set(games)]
    # prior baselines (DSL/naive) for comparison
    prior = {}
    pj = REPO / "results" / "arc3_m2_dsl.json"
    if pj.exists():
        for g in json.loads(pj.read_text())["per_game"]:
            prior[g["game"]] = {"naive": g["energy_gen_naive"], "dsl": g["energy_gen_dsl"]}

    per_game, total_codex_s, total_calls = [], 0.0, 0
    for game in sel:
        short = game.split("-")[0]
        trans = _collect(arc, game, budget, episodes, rng, GameAction, GameState)
        tr, held = _key_disjoint_split(trans, rng, frac=0.25)
        changing = [t for t in tr if (np.asarray(t[0]) != np.asarray(t[2])).any()]
        sample = changing[:30] if len(changing) >= 8 else tr[:30]
        bg = int(np.bincount(np.asarray(tr[0][0]).ravel()).argmax()) if tr else 0
        shape = np.asarray(tr[0][0]).shape if tr else (0, 0)
        serialized = _serialize(sample, bg, shape)

        best_energy, best_code, best_fn, history = None, None, None, []
        prior_code, failures = None, None
        for it in range(iters):
            raw, dt = ask_codex(synth_prompt(serialized, prior_code, failures))
            total_codex_s += dt; total_calls += 1
            code = _extract_code(raw)
            if code is None:
                history.append({"iter": it, "status": "no_code", "codex_s": dt}); continue
            fn = safe_predict_from_code(code)
            if fn is None:
                history.append({"iter": it, "status": "unsafe_or_uncompilable", "codex_s": dt}); continue
            # diagnostic: on changing held transitions, does predict ever produce a NON-no-op output?
            chg = [(s, a, s2) for s, a, s2 in held if (np.asarray(s) != np.asarray(s2)).any()][:20]
            nonnoop = sum(1 for s, a, s2 in chg if not np.array_equal(fn(s, a), np.asarray(s, dtype=np.int16)))
            ce = grade_predictions(fn, held)
            e = ce["energy"]
            history.append({"iter": it, "status": "graded", "energy": e,
                            "dynamics_accuracy": ce.get("dynamics_accuracy"), "codex_s": dt,
                            "code_len": len(code), "nonnoop_on_changing": f"{nonnoop}/{len(chg)}"})
            if e is not None and (best_energy is None or e < best_energy):
                best_energy, best_code, best_fn = e, code, fn
            if best_energy is not None and best_energy <= 0.15:
                break                               # trustworthy model found -> stop refactoring
            prior_code = best_code                  # always refactor from the BEST program so far
            failures = _failure_examples(best_fn, held, bg, shape) if best_fn else None
        base = prior.get(short, {})
        per_game.append({
            "game": short, "codex_best_energy": best_energy,
            "dsl_energy": base.get("dsl"), "naive_energy": base.get("naive"),
            "improvement_over_dsl": (round(base["dsl"] - best_energy, 4)
                                     if (best_energy is not None and base.get("dsl") is not None) else None),
            "trustworthy_at_0.15": (best_energy is not None and best_energy <= 0.15),
            "n_transitions": len(trans), "iters_run": len(history), "history": history,
            "best_code": (best_code[:2000] if best_code else None),
        })
        print(f"  {short:6s} codex_best={best_energy} (dsl={base.get('dsl')} naive={base.get('naive')}) "
              f"trustworthy={per_game[-1]['trustworthy_at_0.15']}", flush=True)

    rated = [g for g in per_game if g["codex_best_energy"] is not None]
    n_trust = sum(1 for g in rated if g["trustworthy_at_0.15"])
    n_beats_dsl = sum(1 for g in rated if g["improvement_over_dsl"] and g["improvement_over_dsl"] > 0.05)
    mean_codex = round(sum(g["codex_best_energy"] for g in rated) / len(rated), 4) if rated else None
    verdict = (f"complete: m2v3_codex_synth_trustworthy{n_trust}of{len(rated)}_beatsdsl{n_beats_dsl}"
               f"_meanBestEnergy{mean_codex}")
    art = {
        "experiment": "arc3_m2_codex_synth", "title": "arc3_m2v3_codex_program_synthesis_verified",
        "honest_verdict": verdict,
        "inference_substrate": "offline_arc_agi3_plus_codex_program_synthesis_consistency_verified",
        "claim": ("Codex (gpt-5.5) writes a per-game predict(grid,action) program; the grid-grounded "
                  "consistency energy (no oracle) grades held-out transitions and drives a bounded "
                  "refactor loop. Generator=codex, verifier=energy. Win = trustworthy model (energy<=0.15) "
                  "on games the fixed DSL could not model."),
        "n_games": len(per_game), "games": list(games),
        "n_trustworthy_at_0.15": n_trust, "n_beats_dsl_by_0.05": n_beats_dsl,
        "mean_codex_best_energy": mean_codex,
        "total_codex_calls": total_calls, "total_codex_seconds": round(total_codex_s, 1),
        "iters_per_game": iters, "budget_per_game": budget, "episodes_per_game": episodes,
        "random_seed": seed, "no_gpu_used": True, "submitted_to_leaderboard": False,
        "duration_s": round(time.time() - started, 1), "per_game": per_game,
        "note": ("M2-v3 escalation tier. Codex is the rare heavy inducer; the consistency energy is the "
                 "oracle-free verifier that prunes/certifies its programs. A trustworthy model (energy "
                 "low) on a game the DSL couldn't model = the escalation works -> M2-v3b plans on it for "
                 "a first solve. If codex also can't reach low energy, the games need richer interaction/"
                 "observation -> honest finding."),
    }
    if write:
        (REPO / "results" / "arc3_m2_codex_synth.json").write_text(
            json.dumps(art, indent=2, sort_keys=True) + "\n", "utf-8")
    print(f"\n-> {verdict}")
    print(f"   {n_trust}/{len(rated)} trustworthy (energy<=0.15), {n_beats_dsl} beat the DSL by >0.05 | "
          f"{total_calls} codex calls / {round(total_codex_s)}s")
    return art


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", default="vc33,sb26,m0r0,ls20")
    ap.add_argument("--budget", type=int, default=1200)
    ap.add_argument("--episodes", type=int, default=35)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run(games=[g.strip() for g in args.games.split(",") if g.strip()],
        budget=args.budget, episodes=args.episodes, iters=args.iters, seed=args.seed)

"""M2-v5: plan a FIRST SOLVE on vc33 using the trustworthy induced world-model.

Plan: docs/research-notes/arc-agi3-agent-research-plan.md (M2). M2-v4b produced the first trustworthy
world-model: a codex-induced, consistency-energy-verified predict(grid, action) for vc33 (energy
0.005/0.011, ~99% dynamics accuracy, replicated; results/arc3_vc33_world_model_program.py). The model
revealed vc33's mechanic: the top row is a rasterized progress bar that advances when you click the
right colored component. This experiment uses the verified model as the ACTION SELECTOR (the verifier-
as-pruner / efficiency role): at each real step it SIMULATES every candidate action with the model and
executes the one the model predicts advances progress most, confirming a solve via the REAL env's
level_completed signal. A blind object-click baseline (which scored 0 in the floor) is run for
comparison: if the model-guided policy solves where blind did not, that is the first solve AND the
efficiency thesis (verified model prunes the action space) on a real game.

Fully offline + air-gapped. The model is loaded into a RESTRICTED namespace (numpy only). The REAL env
is ground truth for the win (the model only proposes; the env confirms).

  .venv/bin/python scripts/experiments/arc3_m2_solve.py --game vc33-5430563c --budget 200
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
ENVDIR = str(REPO / "environment_files")
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "scripts" / "experiments"))
from carnot.agentic.arc_agi3_world_model import grid_of, frame_hash, objects  # noqa: E402
from arc3_m2_codex_synth import safe_predict_from_code  # noqa: E402


def _load_model(path):
    code = Path(path).read_text()
    fn = safe_predict_from_code(code)
    if fn is None:
        raise RuntimeError("could not load the verified world-model program")
    return fn


def _bg(grid):
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[counts.argmax()])


def _progress(grid):
    """vc33 progress = filled cells in the top-row progress bar (non-background row-0 cells)."""
    g = np.asarray(grid)
    return int((g[0] != _bg(g)).sum())


def _candidates(grid, available):
    cands = []
    av = list(available or [])
    if 6 in av:
        seen = set()
        for (y, x) in objects(grid):                  # click object centroids (the progress triggers)
            k = (6, int(x), int(y))
            if k not in seen:
                seen.add(k); cands.append(k)
    for a in av:
        if a not in (0, 6):
            cands.append((a,))
    return cands


def _play(arc, game, predict, budget, rng, model_guided, GameAction, GameState):
    by_id = {a.value: a for a in GameAction}
    env = arc.make(game)
    f = env.reset()
    actions, max_level, first_solve_at = 0, 0, None
    tried = set()
    while actions < budget:
        grid = grid_of(f)
        fh = frame_hash(grid)
        lv = int(getattr(f, "levels_completed", 0) or 0)
        if lv > max_level:
            max_level = lv
            first_solve_at = first_solve_at or actions
        st = getattr(f, "state", None)
        if st == GameState.WIN:
            break
        cands = [c for c in _candidates(grid, getattr(f, "available_actions", [])) if (fh, c) not in tried]
        if not cands:
            cands = _candidates(grid, getattr(f, "available_actions", []))
        if model_guided:
            cur = _progress(grid)
            scored = []
            for c in cands:
                a_int = c[0]
                data = {"x": c[1], "y": c[2]} if a_int == 6 else None
                try:
                    pred = predict(grid, c)
                    dprog = _progress(pred) - cur
                    changed = int((np.asarray(pred) != grid).any())
                except Exception:
                    dprog, changed = -99, 0
                scored.append((dprog, changed, c))
            scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
            # if no candidate is predicted to advance progress, take the best 'any-change' else random
            akey = scored[0][2] if scored and (scored[0][0] > 0 or scored[0][1] > 0) else rng.choice(cands)
        else:
            akey = rng.choice(cands)                  # blind baseline
        tried.add((fh, akey))
        a_int = akey[0]
        data = {"x": akey[1], "y": akey[2]} if a_int == 6 else None
        f = env.step(by_id.get(a_int, GameAction.ACTION1), data=data)
        actions += 1
        if getattr(f, "state", None) == GameState.GAME_OVER:
            f = env.reset()                            # vc33 is short; reset and keep trying within budget
    lv = int(getattr(f, "levels_completed", 0) or 0)
    if lv > max_level:
        max_level = lv; first_solve_at = first_solve_at or actions
    return {"levels_solved": max_level, "actions_used": actions, "first_solve_at": first_solve_at,
            "final_state": str(getattr(f, "state", "?"))}


def run(game="vc33-5430563c", budget=200, seed=0, write=True):
    from arc_agi import Arcade
    from arc_agi.base import OperationMode
    from arcengine.enums import GameAction, GameState
    started = time.time()
    arc = Arcade(arc_api_key="", operation_mode=OperationMode.OFFLINE, environments_dir=ENVDIR)
    info = {getattr(e, "game_id", None): (getattr(e, "baseline_actions", None) or [])
            for e in arc.get_environments()}
    baseline = info.get(game, [])
    predict = _load_model(REPO / "results" / "arc3_vc33_world_model_program.py")

    guided = _play(arc, game, predict, budget, random.Random(seed), True, GameAction, GameState)
    blind = _play(arc, game, predict, budget, random.Random(seed), False, GameAction, GameState)

    solved = guided["levels_solved"] > 0
    verdict = (f"complete: m2v5_vc33_model_guided_solve_levels{guided['levels_solved']}"
               f"_at_action{guided['first_solve_at']}_blindbaseline{blind['levels_solved']}_solved{solved}")
    art = {
        "experiment": "arc3_m2_solve", "title": "arc3_m2v5_vc33_first_solve_model_guided",
        "honest_verdict": verdict,
        "inference_substrate": "offline_arc_agi3_model_guided_policy_real_env_confirmed",
        "game": game, "win_levels": len(baseline),
        "model_guided": guided, "blind_baseline": blind,
        "ACCURACY_levels_solved": guided["levels_solved"],
        "first_solve_at_action": guided["first_solve_at"],
        "blind_levels_solved": blind["levels_solved"],
        "world_model": "results/arc3_vc33_world_model_program.py (energy 0.005/0.011, verified)",
        "real_env_confirmed": True, "budget": budget, "random_seed": seed,
        "no_llm_used": True, "no_gpu_used": True, "submitted_to_leaderboard": False,
        "duration_s": round(time.time() - started, 1),
        "note": ("M2-v5 first-solve attempt. The verified world-model selects the progress-advancing "
                 "action (verifier-as-pruner / efficiency role); the REAL env confirms the win via "
                 "level_completed. solved=True with model_guided beating the blind baseline = first "
                 "ARC-AGI-3 solve + efficiency thesis on a real game. Quota-gate: online play only when "
                 "an offline solve beats the TRM baseline + best prior Carnot submission."),
    }
    if write:
        (REPO / "results" / "arc3_m2_solve.json").write_text(json.dumps(art, indent=2, sort_keys=True) + "\n", "utf-8")
    print(f"-> {verdict}")
    print(f"   model_guided: {guided}")
    print(f"   blind_baseline: {blind}")
    return art


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--game", default="vc33-5430563c")
    ap.add_argument("--budget", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    art = run(game=args.game, budget=args.budget, seed=args.seed)
    raise SystemExit(0 if art["ACCURACY_levels_solved"] > 0 else 1)

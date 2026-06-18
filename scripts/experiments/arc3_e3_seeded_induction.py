"""SEEDED-INDUCTION PROTOTYPE -- does feeding the E3 induce dataset our BANKED WINNING
transitions fix the two floor-stage failures the diagnostic found, and move live E3 off
its 0/6 floor?

The diagnostic (results/arc3_e3_pipeline_diagnostic.json) found the E3 induce->verify->plan
pipeline broken at the FRONT: on cn04 the 80-transition random explore saw ZERO wins, so
the induce prompt had NO win state (see _transitions_block: it only includes a WIN STATE if
explore observed level_after>level_before), so the induced is_level_complete could not fire
on a real win, so the planner could never target one. We already OWN the fix: 34 banked,
frame-only WINNING trajectories. This prototype harvests the win-bearing transitions from a
game's banked solve and SEEDS them into the induce dataset -- a strictly-additive change that
gives the model the positive examples random explore never reached.

A/B per floor game (cn04, ar25), SAME proposer held constant (local gemma-12B GGUF on the
ISOLATED port 8920, matching the diagnostic), measuring every stage the diagnostic did:

  dataset_wins   : # win transitions in the induce dataset      (baseline ~0; seeded >=1 by construction)
  engine_acc     : WorldModelVerifier accuracy on real transitions
  win_pred_fires : does the induced is_level_complete fire on a TRUE banked win grid?
  plan_found     : does plan_in_model return a path (induced win-predicate)?
  plan_grounded  : does plan_in_model find a path with a PROPOSER-INDEPENDENT grounded
                   win-predicate (exact-match vs the banked win grids)? -- isolates whether
                   the win-predicate or the engine is the remaining wall.
  exec_levelup   : if a plan is found, execute it in the REAL env -- did it reach a level-up?
                   (the honest 'live moved off the floor' headline.)

ISOLATION: induced models are written to results/arc_e3_seedproto/<game>/ -- NEVER the
conductor's results/arc_e3/<game>/ (PHASE B2 is actively editing those). Zero quota
(offline sim + local GGUF). Self-cleans its llama-server on exit.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

from arcengine import GameAction
from carnot.agentic import arc_solver_kit as kit
from carnot.agentic import arc_executable_world_model as e3
from carnot.agentic.arc_agi3_world_model import grid_of
from carnot.agentic.arc_agi3_live_adapter import _levels_completed, _game_action

GAMES = ["cn04", "ar25"]          # the gap-1 floor games the diagnostic dissected
EXPLORE_N = 80
PROTO_DIR = REPO / "results" / "arc_e3_seedproto"


def _mh():
    spec = importlib.util.spec_from_file_location(
        "mh", str(REPO / "scripts" / "arc3_replay_scorecard_metaharness.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _banked_actions(game: str):
    """The banked solve's normalized (action_id, data) list + whether the game warms up."""
    mh = _mh()
    src = mh.RESOLVED_ARTIFACTS.get(game, mh.GAME_ARTIFACTS.get(game))
    acts = [a for a in (mh.load_actions(src) or []) if mh.normalize(a)[0] is not None]
    return [mh.normalize(a) for a in acts], (game in mh.WARMUP_GAMES)


def banked_transitions(game: str, cell: int):
    """Replay the banked solve and record logical-resolution transitions -- these CONTAIN the
    win transition(s) (level_after>level_before) random explore never reached, plus the on-path
    transitions leading to each win. Also returns the list of true win grids (for grounding)."""
    acts, warm = _banked_actions(game)
    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    f = env.reset()
    if warm and acts:
        aid, data = acts[0]
        f = env.step(_game_action(GameAction, aid), data=data)
    trans, win_grids = [], []
    for aid, data in acts:
        g0 = e3.to_logical(grid_of(f), cell)
        l0 = _levels_completed(f)
        nf = env.step(_game_action(GameAction, aid), data=data)
        if nf is None:
            break
        g1 = e3.to_logical(grid_of(nf), cell)
        l1 = _levels_completed(nf)
        trans.append(e3.Transition(g0, int(aid), data, g1, l0, l1))
        if l1 > l0:
            win_grids.append(g1)
        f = nf
    return trans, win_grids, warm


def collect_seeded(game: str, n: int):
    """explore-only transitions + banked win-bearing transitions (appended so the win states
    are present for _transitions_block). Returns (transitions, cell, win_grids, warm)."""
    explore, cell = e3.collect_transitions(game, n=n)
    banked, win_grids, warm = banked_transitions(game, cell)
    return explore + banked, cell, win_grids, warm


def _load_engine_from(path: Path):
    spec = importlib.util.spec_from_file_location(f"proto_wm_{path.parent.name}", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return getattr(mod, "engine"), getattr(mod, "is_level_complete", None)


def induce_to_proto(proposer, game: str, trans, cell: int):
    """Build the induce prompt, get code via generate() (does NOT write to disk), write it to
    the ISOLATED proto dir, and load it. Returns (ok, msg, engine, is_done)."""
    prompt = (e3.induce_prompt(game, trans, cell) +
              "\n\nReturn ONLY one ```python code block with engine + is_level_complete.\n```python\n")
    ok, code = proposer.generate(prompt, ("engine", "is_level_complete"))
    if not ok:
        return False, str(code)[:160], None, None
    d = PROTO_DIR / game
    d.mkdir(parents=True, exist_ok=True)
    (d / "world_model.py").write_text(code)
    try:
        engine, is_done = _load_engine_from(d / "world_model.py")
    except Exception as ex:
        return False, f"load failed: {type(ex).__name__}: {str(ex)[:120]}", None, None
    return True, "wrote+loaded proto world_model.py", engine, is_done


def _grounded_win_pred(win_grids):
    """A proposer-INDEPENDENT win predicate: True iff the grid exactly matches an observed
    banked win grid. By construction fires on a true win -> isolates whether the win-predicate
    or the engine is the remaining wall when used in plan_in_model with the induced engine."""
    wins = [np.asarray(w) for w in win_grids]

    def is_done(grid):
        g = np.asarray(grid)
        return any(g.shape == w.shape and np.array_equal(g, w) for w in wins)
    return is_done


def _root_grid(game: str, cell: int, warm: bool):
    from carnot.agentic.arc_graph_explore import _warm
    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    f = _warm(env, warm)
    return e3.to_logical(grid_of(f), cell)


def execute_plan(game: str, plan, warm: bool) -> int:
    """Execute a planned action sequence in the REAL env from reset(+warmup); return the number
    of level-ups achieved (the honest 'did the induced plan actually work live' check)."""
    from carnot.agentic.arc_graph_explore import _warm
    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    f = _warm(env, warm)
    l0 = _levels_completed(f)
    lvl = l0
    for step in plan:
        f = env.step(_game_action(GameAction, step["action"]), data=step["data"])
        if f is None:
            break
        lvl = max(lvl, _levels_completed(f))
    return max(0, lvl - l0)


def run_arm(proposer, game, trans, cell, true_win, win_grids, warm, *, label):
    rec = {"label": label, "dataset_n": len(trans),
           "dataset_wins": sum(1 for t in trans if t.level_after > t.level_before)}
    ok, msg, engine, is_done = induce_to_proto(proposer, game, trans, cell)
    rec["induce"] = {"ok": ok, "msg": msg}
    if not ok:
        rec["verdict"] = "INDUCE_FAILED"
        return rec
    vr = e3.WorldModelVerifier(trans).score(engine)
    rec["engine_acc"] = round(vr.accuracy, 3)
    # win predicate on a TRUE win grid
    wp = None
    if true_win is not None and is_done is not None:
        try:
            wp = bool(is_done(true_win))
        except Exception:
            wp = None
    rec["win_pred_fires"] = wp
    root = _root_grid(game, cell, warm)
    # plan with the INDUCED win-predicate
    plan = e3.plan_in_model(engine, is_done, root)
    rec["plan_found"] = plan is not None
    # plan with a PROPOSER-INDEPENDENT grounded win-predicate (isolates engine vs win-pred)
    gplan = e3.plan_in_model(engine, _grounded_win_pred(win_grids), root)
    rec["plan_grounded"] = gplan is not None
    # execute whichever plan exists in the REAL env -> did it reach a level-up?
    chosen = plan or gplan
    rec["exec_levelup"] = execute_plan(game, chosen, warm) if chosen else 0
    if rec["dataset_wins"] == 0:
        rec["verdict"] = "EXPLORE_SAW_NO_WIN"
    elif vr.accuracy < 0.5:
        rec["verdict"] = f"ENGINE_INACCURATE_{vr.accuracy:.0%}"
    elif wp is False:
        rec["verdict"] = "WIN_PREDICATE_WRONG"
    elif not (plan or gplan):
        rec["verdict"] = "PLAN_NOT_FOUND"
    elif rec["exec_levelup"] > 0:
        rec["verdict"] = "LIVE_LEVELUP"
    else:
        rec["verdict"] = "PLAN_FOUND_BUT_EXEC_DIVERGED"
    return rec


def diagnose(proposer, game: str) -> dict:
    cell0 = e3.detect_cell(grid_of(kit.offline_arcade().make(
        game, scorecard_id=kit.offline_arcade().open_scorecard()).reset()))
    true_win = _true_win_grid(game, cell0)
    base_trans, base_cell = e3.collect_transitions(game, n=EXPLORE_N)
    _, base_wins, base_warm = banked_transitions(game, base_cell)   # win grids for grounding only
    seed_trans, seed_cell, seed_wins, warm = collect_seeded(game, EXPLORE_N)
    baseline = run_arm(proposer, game, base_trans, base_cell, true_win, base_wins, base_warm,
                       label="baseline_explore_only")
    seeded = run_arm(proposer, game, seed_trans, seed_cell, true_win, seed_wins, warm,
                     label="seeded_explore_plus_banked_wins")
    return {"game": game, "true_win_available": true_win is not None,
            "baseline": baseline, "seeded": seeded,
            "seeding_moved_a_stage": _moved(baseline, seeded)}


def _moved(b: dict, s: dict) -> dict:
    """Did seeding move any stage off its floor?"""
    return {
        "dataset_wins": f"{b.get('dataset_wins')}->{s.get('dataset_wins')}",
        "win_pred_fires": f"{b.get('win_pred_fires')}->{s.get('win_pred_fires')}",
        "plan_found": f"{b.get('plan_found')}->{s.get('plan_found')}",
        "plan_grounded": f"{b.get('plan_grounded')}->{s.get('plan_grounded')}",
        "exec_levelup": f"{b.get('exec_levelup')}->{s.get('exec_levelup')}",
        "verdict": f"{b.get('verdict')}->{s.get('verdict')}",
    }


def _true_win_grid(game: str, cell: int):
    """Replay the banked solution to the FIRST level-up; return the logical win grid (or None)."""
    acts, warm = _banked_actions(game)
    if not acts:
        return None
    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    f = env.reset()
    if warm and acts:
        aid, data = acts[0]
        f = env.step(_game_action(GameAction, aid), data=data)
    l0 = _levels_completed(f)
    for aid, data in acts:
        f = env.step(_game_action(GameAction, aid), data=data)
        if f is None:
            return None
        if _levels_completed(f) > l0:
            return e3.to_logical(grid_of(f), cell)
    return None


def main() -> int:
    # --games cn04[,ar25] subset; --timeout S (iGPU is ~10x slower than a 3090, so the default
    # 300s urlopen often cuts off a full 4096-token world-model mid-generation -> 900s here).
    argv = sys.argv[1:]
    games = (argv[argv.index("--games") + 1].split(",") if "--games" in argv else GAMES)
    timeout = int(argv[argv.index("--timeout") + 1]) if "--timeout" in argv else 900
    print(f"== E3 SEEDED-INDUCTION prototype: does seeding the induce dataset with banked wins "
          f"move live E3 off 0? (games={games} timeout={timeout}s) ==", flush=True)
    proposer = e3.LocalGGUFProposer(repo_substr="gemma-4-12B-it", port=8920, timeout=timeout)
    rows = []
    try:
        for g in games:
            r = diagnose(proposer, g)
            rows.append(r)
            print(f"\n  [{g}] seeding moved: {json.dumps(r['seeding_moved_a_stage'])}", flush=True)
            print(f"    baseline: {json.dumps({k: r['baseline'].get(k) for k in ('dataset_wins','engine_acc','win_pred_fires','plan_found','plan_grounded','exec_levelup','verdict')})}", flush=True)
            print(f"    seeded:   {json.dumps({k: r['seeded'].get(k) for k in ('dataset_wins','engine_acc','win_pred_fires','plan_found','plan_grounded','exec_levelup','verdict')})}", flush=True)
    finally:
        try:
            proposer.stop()
        except Exception:
            pass
    any_levelup = any(r["seeded"].get("exec_levelup", 0) > 0 for r in rows)
    any_stage_moved = any(
        r["seeded"].get("dataset_wins", 0) > r["baseline"].get("dataset_wins", 0)
        or (r["seeded"].get("win_pred_fires") and not r["baseline"].get("win_pred_fires"))
        or (r["seeded"].get("plan_grounded") and not r["baseline"].get("plan_grounded"))
        for r in rows)
    verdict = ("complete_seeded_induction_live_levelup" if any_levelup
               else "complete_seeded_induction_stage_moved_no_levelup" if any_stage_moved
               else "complete_seeded_induction_no_change")
    out = {"experiment": "arc3_e3_seeded_induction", "games": games,
           "per_game": rows, "any_live_levelup": any_levelup, "any_stage_moved": any_stage_moved,
           "honest_verdict": verdict,
           "inference_substrate": "offline_sim_no_quota_e3_local_gguf_induction_port8920"}
    (REPO / "results" / "arc3_e3_seeded_induction.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\n  any live level-up: {any_levelup} | any stage moved: {any_stage_moved}", flush=True)
    print(f"  -> {verdict}", flush=True)
    print(f"  wrote results/arc3_e3_seeded_induction.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

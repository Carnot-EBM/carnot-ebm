"""LEAVE-ONE-GAME-OUT INDUCTION-TRANSFER PROBE -- does anything we learn OFFLINE make us
better at modeling the dynamics of a game we have NEVER seen?

The operator's question: a per-game dynamics engine pre-trained on game X is useless on an
unseen hidden game -- so how does the offline work help live? This probe answers the PREMISE
empirically and proposer-free: does cross-game dynamics STRUCTURE even exist to transfer?

For every ordered pair (X -> G) of solved games, we score game X's induced engine against
game G's REAL transitions (WorldModelVerifier accuracy). The diagonal (X==G) is each engine's
self-accuracy (source quality). The off-diagonal is whole-engine transfer. Two honest baselines:
  IDENTITY  : predict no change (next==grid). Many candidate actions are no-ops, so identity
              scores high on the FULL set -- so we also measure CHANGED-only accuracy.
  CHANGED   : accuracy on transitions where the grid actually changed AND no level-up occurred
              (exclude win discontinuities per the seeded-induction finding) -- the REAL
              dynamics-prediction signal. Transfer is real only if cross-game CHANGED-acc beats
              identity's CHANGED-acc (~0 -- identity never predicts a change).

LOGO read per held-out game G: self_changed (can our own engine model G?), best cross-game
CHANGED-acc + which game it came from (the transfer ceiling), vs identity. If the transfer
ceiling is ~0 across games, WHOLE-ENGINE reuse does NOT transfer -> the offline->live bridge
must live at the PRIMITIVE/skill level (a cross-game inducer or a mechanic-prior library), not
engine reuse. If some pairs transfer, they name the shared mechanics worth a primitive library.

This parallels the cross-game VALUE-head LOGO (which came up CHANCE, 0.514). Proposer-free,
zero quota, CPU. Engines are SNAPSHOT-copied first so a conductor mid-edit can't corrupt a read.
"""
from __future__ import annotations

import importlib.util
import json
import shutil
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

GAMES = ["ar25", "cn04", "ft09", "ka59", "sc25", "tr87"]   # all load an engine; transition-sourceable
EXPLORE_N = 80
SNAP = REPO / "results" / "arc_logo_snapshot"


def _mh():
    spec = importlib.util.spec_from_file_location(
        "mh", str(REPO / "scripts" / "arc3_replay_scorecard_metaharness.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def snapshot_engines(games):
    """Copy each game's results/arc_e3/<g>/world_model.py to an isolated snapshot dir and load it
    from there -- a consistent read that a conductor mid-edit cannot corrupt. Returns {game: engine}."""
    engines = {}
    for g in games:
        src = REPO / "results" / "arc_e3" / g / "world_model.py"
        if not src.exists():
            continue
        d = SNAP / g
        d.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copyfile(src, d / "world_model.py")
            spec = importlib.util.spec_from_file_location(f"snap_wm_{g}", str(d / "world_model.py"))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            engines[g] = getattr(mod, "engine")
        except Exception as ex:
            print(f"  [snapshot] {g}: engine load failed -> excluded as source ({type(ex).__name__})", flush=True)
    return engines


def banked_transitions(game: str, cell: int):
    """Replay the banked solve (if any) -> logical-resolution transitions (incl wins)."""
    mh = _mh()
    src = mh.RESOLVED_ARTIFACTS.get(game, mh.GAME_ARTIFACTS.get(game))
    if not src:                                    # e.g. ft09 has no banked-solve source -> explore-only
        return []
    acts = [mh.normalize(a) for a in (mh.load_actions(src) or []) if mh.normalize(a)[0] is not None]
    if not acts:
        return []
    warm = game in mh.WARMUP_GAMES
    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    f = env.reset()
    if warm:
        aid, data = acts[0]
        f = env.step(_game_action(GameAction, aid), data=data)
    trans = []
    for aid, data in acts:
        g0 = e3.to_logical(grid_of(f), cell)
        l0 = _levels_completed(f)
        nf = env.step(_game_action(GameAction, aid), data=data)
        if nf is None:
            break
        trans.append(e3.Transition(g0, int(aid), data, e3.to_logical(grid_of(nf), cell),
                                    l0, _levels_completed(nf)))
        f = nf
    return trans


def collect(game: str):
    """Frame-only transitions for a game: explore + banked. Returns (transitions, cell)."""
    explore, cell = e3.collect_transitions(game, n=EXPLORE_N)
    return explore + banked_transitions(game, cell), cell


def _changed_mask(trans):
    """Transitions where the grid actually changed AND no level-up (exclude win discontinuities)."""
    return [t for t in trans
            if t.level_after == t.level_before and not np.array_equal(t.grid, t.next_grid)]


def score(engine, trans) -> float:
    """Exact-match accuracy of engine over a transition list (crash/shape-mismatch -> wrong)."""
    if not trans:
        return float("nan")
    ok = 0
    for t in trans:
        try:
            pred = np.asarray(engine(t.grid.copy(), t.action, t.data))
        except Exception:
            continue
        if pred.shape == t.next_grid.shape and np.array_equal(pred, t.next_grid):
            ok += 1
    return ok / len(trans)


def _identity(grid, action, data):
    return grid


def main() -> int:
    print(f"== LEAVE-ONE-GAME-OUT induction-transfer probe (games={GAMES}) ==", flush=True)
    engines = snapshot_engines(GAMES)
    sources = sorted(engines)
    print(f"  engine sources that loaded: {sources}", flush=True)

    # collect each game's transitions once
    trans_all, changed_all = {}, {}
    for g in GAMES:
        tr, cell = collect(g)
        trans_all[g] = tr
        changed_all[g] = _changed_mask(tr)
        print(f"  [{g}] transitions={len(tr)} changed(dynamics)={len(changed_all[g])} cell={cell}", flush=True)

    # cross matrix on the CHANGED (real-dynamics) subset + identity baseline
    rows = []
    for G in GAMES:
        chg = changed_all[G]
        identity_changed = score(_identity, chg)          # ~0 expected (identity never predicts change)
        identity_overall = score(_identity, trans_all[G])  # high if many no-ops
        self_changed = score(engines[G], chg) if G in engines else float("nan")
        cross = {}
        for X in sources:
            if X == G:
                continue
            cross[X] = round(score(engines[X], chg), 3)
        best_src = max(cross, key=cross.get) if cross else None
        best_cross = cross.get(best_src, float("nan")) if best_src else float("nan")
        rows.append({
            "held_out_game": G,
            "n_changed": len(chg),
            "identity_changed_acc": round(identity_changed, 3),
            "identity_overall_acc": round(identity_overall, 3),
            "self_changed_acc": round(self_changed, 3) if self_changed == self_changed else None,
            "best_cross_changed_acc": round(best_cross, 3) if best_cross == best_cross else None,
            "best_cross_source": best_src,
            "transfer_gain_over_identity": (round(best_cross - identity_changed, 3)
                                            if best_cross == best_cross else None),
            "cross_matrix_changed": cross,
        })
        print(f"\n  HELD-OUT {G}: self_changed={rows[-1]['self_changed_acc']} | "
              f"best_cross_changed={rows[-1]['best_cross_changed_acc']} (from {best_src}) | "
              f"identity_changed={rows[-1]['identity_changed_acc']} | "
              f"transfer_gain={rows[-1]['transfer_gain_over_identity']}", flush=True)

    # --- FOCUSED SIMILAR-PAIR test (operator's design: pick two similar games; does learning A
    #     help model B, controlling for similarity?). --pair A,B ; default ar25,ka59 (both are
    #     E3 'select an object and move/push it to a goal' solves -- same mechanic, different layout)
    argv = sys.argv[1:]
    pair = (argv[argv.index("--pair") + 1].split(",") if "--pair" in argv else ["ar25", "ka59"])
    pair_focus = None
    if len(pair) == 2 and all(p in GAMES for p in pair):
        a, b = pair
        by_game = {r["held_out_game"]: r for r in rows}

        def _transfer(src, tgt):                       # src engine -> tgt's changed dynamics
            return by_game[tgt]["cross_matrix_changed"].get(src)

        def _dissimilar_control(tgt):                  # mean transfer into tgt from all NON-partner sources
            cm = by_game[tgt]["cross_matrix_changed"]
            other = [v for k, v in cm.items() if k not in pair]
            return round(float(np.mean(other)), 3) if other else None

        a_to_b, b_to_a = _transfer(a, b), _transfer(b, a)
        ctrl_b, ctrl_a = _dissimilar_control(b), _dissimilar_control(a)
        pair_focus = {
            "pair": pair, "rationale": "both E3 select-and-move-object-to-goal solves (same mechanic class)",
            f"{a}_to_{b}_changed_acc": a_to_b, f"{b}_to_{a}_changed_acc": b_to_a,
            f"{b}_self_changed_acc": by_game[b]["self_changed_acc"],
            f"{a}_self_changed_acc": by_game[a]["self_changed_acc"],
            f"{b}_identity_changed_acc": by_game[b]["identity_changed_acc"],
            f"{a}_identity_changed_acc": by_game[a]["identity_changed_acc"],
            f"dissimilar_control_into_{b}": ctrl_b, f"dissimilar_control_into_{a}": ctrl_a,
            "similar_pair_beats_dissimilar": bool(
                (a_to_b is not None and ctrl_b is not None and a_to_b > ctrl_b + 0.05)
                or (b_to_a is not None and ctrl_a is not None and b_to_a > ctrl_a + 0.05)),
        }
        print(f"\n  == FOCUSED SIMILAR-PAIR {a}<->{b} ==", flush=True)
        print(f"    {a}->{b} changed_acc={a_to_b} (vs dissimilar-control {ctrl_b}, {b}_self={pair_focus[f'{b}_self_changed_acc']})", flush=True)
        print(f"    {b}->{a} changed_acc={b_to_a} (vs dissimilar-control {ctrl_a}, {a}_self={pair_focus[f'{a}_self_changed_acc']})", flush=True)
        print(f"    similar_pair_beats_dissimilar={pair_focus['similar_pair_beats_dissimilar']}", flush=True)

    # headline
    valid_gain = [r["transfer_gain_over_identity"] for r in rows
                  if r["transfer_gain_over_identity"] is not None and r["n_changed"] >= 3]
    mean_gain = round(float(np.mean(valid_gain)), 3) if valid_gain else None
    n_transfer = sum(1 for r in rows if (r["best_cross_changed_acc"] or 0) > 0.5 and r["n_changed"] >= 3)
    self_vals = [r["self_changed_acc"] for r in rows if r["self_changed_acc"] is not None and r["n_changed"] >= 3]
    mean_self = round(float(np.mean(self_vals)), 3) if self_vals else None
    transfer_exists = bool(mean_gain is not None and mean_gain > 0.2) or n_transfer >= 1
    verdict = ("complete_logo_transfer_whole_engine_transfers" if transfer_exists
               else "complete_logo_transfer_no_whole_engine_transfer_bridge_must_be_primitive_level")
    out = {
        "experiment": "arc3_logo_induction_transfer",
        "games": GAMES, "engine_sources": sources,
        "mean_self_changed_acc": mean_self,
        "mean_transfer_gain_over_identity": mean_gain,
        "n_games_with_whole_engine_transfer": n_transfer,
        "whole_engine_transfer_exists": transfer_exists,
        "similar_pair_focus": pair_focus,
        "per_game": rows,
        "interpretation": (
            "self_changed_acc = can our own induced engine model a game's real dynamics. "
            "best_cross_changed_acc = the transfer ceiling: the best OTHER game's engine on this "
            "game's dynamics. transfer_gain = best_cross - identity (identity never predicts a "
            "change, so a positive gain is real whole-engine transfer). If the ceiling is ~identity "
            "across games, whole-engine reuse does NOT transfer and the offline->live bridge must be "
            "a cross-game INDUCER / mechanic-PRIMITIVE library, not engine reuse."),
        "honest_verdict": verdict,
        "inference_substrate": "offline_sim_no_quota_proposer_free_engine_cross_application",
    }
    (REPO / "results" / "arc3_logo_induction_transfer.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\n  mean self_changed={mean_self} | mean transfer_gain_over_identity={mean_gain} | "
          f"games w/ whole-engine transfer={n_transfer}", flush=True)
    print(f"  -> {verdict}", flush=True)
    print(f"  wrote results/arc3_logo_induction_transfer.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

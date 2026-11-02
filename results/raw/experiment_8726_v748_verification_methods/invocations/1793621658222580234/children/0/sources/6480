"""Train the grid-CNN VALUE NET (arc_value_net) -- the higher-capacity successor to the linear value
head that conclusively could not route the live search. Corpus = ALL banked solves, frame-only:

  ON-PATH states  : (grid, normalized steps-to-next-level-up)  -- the winning trajectory, decreasing
                    toward each level-up.
  OFF-PATH negatives: from each on-path state, step a NON-trajectory candidate action -> a reachable
                    off-path grid, labeled (on-path value + a penalty). This is the discrimination
                    signal the linear head lacked: teach the net on-path < off-path so it can ROUTE.

Two outputs: (1) a leave-one-GAME-out DISCRIMINATION check -- on a held-out game, does the net rank
on-path states BELOW their off-path siblings? (the routing signal, measured independent of the live
eval). (2) the full model -> models/arc_value_net.json, loaded by the live explorer. CPU; zero quota.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

from arcengine import GameAction
from carnot.agentic import arc_solver_kit as kit
from carnot.agentic.arc_value_net import ValueNet, _to_grid
from carnot.agentic.arc_agi3_live_adapter import _levels_completed
from carnot.agentic.arc_graph_explore import rich_action_candidates


def _mh():
    spec = importlib.util.spec_from_file_location(
        "mh", str(REPO / "scripts" / "arc3_replay_scorecard_metaharness.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _step(env, aid, data):
    return env.step(getattr(GameAction, f"ACTION{aid}"), data=data)


def collect(off_path: int = 2):
    """Per game: replay the banked solve -> on-path (grid, steps-to-next-level-up); from each on-path
    state branch `off_path` non-trajectory candidates -> off-path grids labeled (value + penalty).
    Returns {game: (grids, values)} so we can do leave-one-game-out. Frame-only."""
    mh = _mh()
    arc = kit.offline_arcade()
    per_game = {}
    for game in sorted(mh.GAME_ARTIFACTS):
        src = mh.RESOLVED_ARTIFACTS.get(game, mh.GAME_ARTIFACTS[game])
        acts = [a for a in (mh.load_actions(src) or []) if mh.normalize(a)[0] is not None]
        if not acts:
            continue
        warm = game in mh.WARMUP_GAMES
        # --- on-path replay: record (grid, level) before each action ---
        env = arc.make(game, scorecard_id=arc.open_scorecard())
        f = env.reset()
        if warm:
            aid, data = mh.normalize(acts[0])
            f = _step(env, aid, data)
        seq = []                                   # (grid, level, frame_for_candidates)
        for a in acts:
            aid, data = mh.normalize(a)
            seq.append((_to_grid(f), _levels_completed(f), f))
            f = _step(env, aid, data)
            if f is None:
                break
        seq.append((_to_grid(f), _levels_completed(f), f))
        # --- label on-path by normalized steps-to-next-level-up ---
        n = len(seq)
        nxt = [None] * n
        d = None
        run = 0
        for i in range(n - 1, -1, -1):
            if i < n - 1 and seq[i + 1][1] > seq[i][1]:
                d, run = 0, 1
            elif d is not None:
                d += 1
                run += 1
            nxt[i] = (d, run)
        grids, vals = [], []
        onpath_val = {}
        for i in range(n):
            if nxt[i] is None or nxt[i][0] is None:
                continue
            v = nxt[i][0] / max(1, nxt[i][1])
            grids.append(seq[i][0])
            vals.append(v)
            onpath_val[i] = v
        # --- off-path negatives: branch non-trajectory candidates from each on-path state ---
        rng = np.random.RandomState(hash(game) & 0xFFFF)
        for i in list(onpath_val)[:-1]:            # not the final (win) state
            onpath_aid, onpath_data = mh.normalize(acts[i]) if i < len(acts) else (None, None)
            cands = rich_action_candidates(seq[i][2])
            alt = [c for c in cands if not (int(c.action_id) == onpath_aid
                                            and (c.data or None) == (onpath_data or None))]
            rng.shuffle(alt)
            for c in alt[:off_path]:
                e2 = arc.make(game, scorecard_id=arc.open_scorecard())
                g = e2.reset()
                if warm:
                    aw, dw = mh.normalize(acts[0])
                    g = _step(e2, aw, dw)
                for a in acts[:i]:                 # replay prefix to state i
                    aid, data = mh.normalize(a)
                    g = _step(e2, aid, data)
                    if g is None:
                        break
                if g is None:
                    continue
                g2 = _step(e2, int(c.action_id), c.data)   # the OFF-path step
                if g2 is None or _levels_completed(g2) > seq[i][1]:
                    continue                       # skip if it actually won (not a negative)
                grids.append(_to_grid(g2))
                vals.append(onpath_val[i] + 0.5)   # penalty: off-path is WORSE than the on-path next state
        per_game[game] = (grids, vals)
    return per_game


def _discrimination(net, grids, vals):
    """Spearman-ish: fraction of (lower-true-value, higher-true-value) pairs the net orders correctly."""
    preds = np.array([net.predict_grid(g) for g in grids])
    tv = np.array(vals)
    order = np.argsort(tv)
    correct = total = 0
    for a in range(len(order)):
        for b in range(a + 1, min(a + 25, len(order))):   # local pairs (cheap)
            ia, ib = order[a], order[b]
            if tv[ia] == tv[ib]:
                continue
            total += 1
            correct += int(preds[ia] <= preds[ib])
    return correct / max(1, total)


def main() -> int:
    print("== train grid-CNN VALUE NET on banked solves (offline->live, higher capacity) ==", flush=True)
    per_game = collect()
    allg = [g for gs, _ in per_game.values() for g in gs]
    allv = [v for _, vs in per_game.values() for v in vs]
    print(f"  corpus: {len(allg)} frame-only states ({sum(len(gs) for gs,_ in per_game.values())} "
          f"incl off-path negatives) from {len(per_game)} games", flush=True)

    # leave-one-GAME-out discrimination: does the net rank progress on a game it NEVER trained on?
    import statistics
    held = []
    for hold in sorted(per_game):
        tr_g = [g for gm, (gs, _) in per_game.items() if gm != hold for g in gs]
        tr_v = [v for gm, (_, vs) in per_game.items() if gm != hold for v in vs]
        hg, hv = per_game[hold]
        if len(hg) < 6:
            continue
        net = ValueNet().fit(tr_g, tr_v, epochs=40)
        acc = _discrimination(net, hg, hv)
        held.append((hold, round(acc, 2)))
        print(f"  LOGO {hold:5}: held-out pair-order accuracy {acc:.2f}", flush=True)
    mean_acc = round(statistics.mean(a for _, a in held), 3) if held else 0.0

    # train the FULL model on everything + save
    net = ValueNet().fit(allg, allv, epochs=80)
    ckpt = REPO / "models" / "arc_value_net.json"
    net.save(ckpt, meta={"trained_games": sorted(per_game), "corpus": len(allg),
                         "label": "normalized_steps_to_next_level_up + off_path_negatives",
                         "logo_mean_pair_accuracy": mean_acc})
    print(f"  saved {ckpt.relative_to(REPO)} | LOGO mean pair-order acc = {mean_acc} "
          f"(0.5=chance; >0.5 = the net learned cross-game routing signal)", flush=True)
    out = {"experiment": "arc_value_net_train", "corpus": len(allg),
           "logo_per_game": held, "logo_mean_pair_accuracy": mean_acc,
           "honest_verdict": (f"complete_value_net_logo_pair_accuracy_{mean_acc}"
                              + ("_learns_cross_game_signal" if mean_acc > 0.55 else "_at_or_near_chance")),
           "checkpoint": "models/arc_value_net.json", "inference_substrate": "offline_cpu_no_quota"}
    (REPO / "results" / "arc_value_net_train.json").write_text(json.dumps(out, indent=2))
    print(f"  {out['honest_verdict']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

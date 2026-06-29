"""Profile the LIVE E3 agent's action budget offline on the games it solves.

WHY: RHAE squares efficiency (per-level min((human/agent)^2*100, 115), index-weighted mean over
levels with UNSOLVED=0). Efficiency on solved levels is a real lever, but we did not know WHERE the
agent's actions go. This splits the per-game action budget into the buckets that map 1:1 to the
efficiency levers, so we target the right knob instead of guessing:

  - discovery_to_first_solve : actions L0->L1 (the exploration/discovery overhead a human barely pays)
  - noop_actions             : actions whose frame did NOT change (pure waste -> the no-op-pruning lever
                               `frame_change_prune_threshold`, currently None/OFF in the submitted config)
  - revisit_actions          : actions returning to an already-seen frame (search redundancy -> value
                               routing / discriminative pruning)
  - per_level marginal cost   : agent actions per level vs the human baseline (exploit-path optimality)

DRIVER: reuses scripts/arc_leaderboard_eval.py:_build_policy('e3') (the FULL competition cascade that
make_carnot_agent runs) + the offline arcade, force-explore from scratch (the unseen-game simulation).
proposer=None (no GGUF) -- the generator affects COVERAGE, not the explore/no-op accounting; declared.
Substrate: verifier_ensemble_against_cached_candidates (offline search, no LLM). No registry writes.
"""

from __future__ import annotations

import importlib.util
import json
import statistics
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULT = REPO / "results" / "experiment_action_budget_profile.json"
BUDGET = 400  # matches the live agent's per-game MAX_ACTIONS
PER_GAME_WALL_S = 240  # guard: skip a game that exceeds this (keeps the whole profile bounded)

_spec = importlib.util.spec_from_file_location("lbe", str(REPO / "scripts" / "arc_leaderboard_eval.py"))
lbe = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lbe)
from carnot.agentic.arc_agi3_world_model import frame_hash, grid_of  # noqa: E402

GAMES = "ar25 bp35 cd82 cn04 dc22 ft09 g50t ka59 lf52 lp85 ls20 m0r0 r11l re86 s5i5 sb26 sc25 sk48 sp80 su15 tn36 tr87 tu93 vc33 wa30".split()


def _log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def profile_game(game: str) -> dict:
    """Drive the e3 policy force-explore; instrument no-op / revisit / per-level cost."""
    policy = lbe._build_policy("e3", game)
    arc = lbe.kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    base = lbe._baseline_actions(env, game)

    frames: list = []
    latest = None
    actions = noop = revisit = 0
    seen: set = set()
    start = best = None
    level_up_actions: list[int] = []
    t0 = time.time()

    for _ in range(BUDGET):
        if time.time() - t0 > PER_GAME_WALL_S:
            break
        try:
            if policy.is_done(frames, latest):
                break
            kind, data = policy.next_move(frames, latest)
        except Exception as e:  # a policy crash on a game is itself a datum
            return {"game": game, "error": f"{type(e).__name__}: {e}", "actions": actions}
        if kind == "RESET":
            latest = env.reset()
        elif kind is None:
            break
        else:
            before = latest
            latest = env.step(getattr(lbe.GameAction, f"ACTION{kind}"), data=data)
            actions += 1
            if latest is not None:
                h = frame_hash(grid_of(latest))
                if before is not None and frame_hash(grid_of(before)) == h:
                    noop += 1  # frame unchanged by this action == no-op
                if h in seen:
                    revisit += 1
                seen.add(h)
        lvl = lbe._level_of(latest)
        if start is None:
            start, best = lvl, lvl
        if best is not None and lvl > best:
            for _lv in range(best, lvl):
                level_up_actions.append(actions)
            best = lvl
        frames.append(latest)
        if latest is None:
            break

    reached = lbe._level_of(latest) if latest is not None else (best or 0)
    levels = max(0, reached - (start or 0))
    # per-level marginal cost vs human baseline
    baseline_list = base.get("per_level") or base.get("levels") or []
    per_level = []
    prev = 0
    for li, at in enumerate(level_up_actions):
        hb = baseline_list[li] if li < len(baseline_list) else None
        per_level.append({"level": li, "agent_actions": at - prev, "human_actions": hb})
        prev = at
    return {
        "game": game,
        "solved_levels": levels,
        "reached_level": reached,
        "total_actions": actions,
        "noop_actions": noop,
        "revisit_actions": revisit,
        "noop_frac": round(noop / actions, 4) if actions else None,
        "revisit_frac": round(revisit / actions, 4) if actions else None,
        "discovery_to_first_solve": level_up_actions[0] if level_up_actions else None,
        "per_level_cost": per_level,
        "wall_s": round(time.time() - t0, 1),
    }


SHARD_DIR = REPO / "results" / "abp_shards"


def main() -> None:
    import sys
    t_start = time.time()
    # --game G : profile ONE game, write a shard file (for parallel sharding).
    if "--game" in sys.argv:
        g = sys.argv[sys.argv.index("--game") + 1]
        SHARD_DIR.mkdir(parents=True, exist_ok=True)
        r = profile_game(g)
        (SHARD_DIR / f"{g}.json").write_text(json.dumps(r))
        _log(f"{g}: solved={r.get('solved_levels')} actions={r.get('total_actions')} "
             f"noop={r.get('noop_frac')} revisit={r.get('revisit_frac')} "
             f"disc1={r.get('discovery_to_first_solve')} {r.get('wall_s')}s"
             + (f" ERR={r['error']}" if r.get('error') else ""))
        return

    # --aggregate : read all shard files, write the summary artifact.
    if "--aggregate" in sys.argv:
        rows = []
        for g in GAMES:
            f = SHARD_DIR / f"{g}.json"
            if f.exists():
                rows.append(json.loads(f.read_text()))
        _finish(rows, t_start)
        return

    # default: run all games sequentially.
    rows = []
    for g in GAMES:
        r = profile_game(g)
        rows.append(r)
        _log(f"{g}: solved={r.get('solved_levels')} actions={r.get('total_actions')} "
             f"noop={r.get('noop_frac')} revisit={r.get('revisit_frac')} "
             f"disc1={r.get('discovery_to_first_solve')} {r.get('wall_s')}s"
             + (f" ERR={r['error']}" if r.get('error') else ""))
    _finish(rows, t_start)


def _finish(rows: list, t_start: float) -> None:
    solved = [r for r in rows if (r.get("solved_levels") or 0) >= 1]

    def _med(key):
        vals = [r[key] for r in solved if r.get(key) is not None]
        return round(statistics.median(vals), 4) if vals else None

    summary = {
        "n_games": len(rows),
        "n_solved_games": len(solved),
        "solved_games": [r["game"] for r in solved],
        "median_total_actions_solved": _med("total_actions"),
        "median_noop_frac_solved": _med("noop_frac"),
        "median_revisit_frac_solved": _med("revisit_frac"),
        "median_discovery_to_first_solve": _med("discovery_to_first_solve"),
        "budget_per_game": BUDGET,
    }

    artifact = {
        "experiment": "action_budget_profile",
        "summary": summary,
        "per_game": rows,
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "solve_provenance": "development_proxy",
        "read_game_source": False,
        "used_env_source": True,
        "proposer": "none_no_gguf",
        "random_seed": 0,
        "duration_s": round(time.time() - t_start, 2),
        "honest_verdict": "complete_action_budget_profiled_offline_proposer_none",
        "methodology_note": (
            "Drove arc_leaderboard_eval._build_policy('e3') (the full competition cascade) force-explore "
            "from scratch on the offline arcade, budget=400 (live MAX_ACTIONS). no-op = frame_hash "
            "unchanged after a step; revisit = frame_hash already seen; discovery_to_first_solve = actions "
            "to L0->L1. proposer=None (no GGUF) -- generator affects coverage not the explore/no-op "
            "accounting; declared. development_proxy; no registry/router/checkpoint writes."
        ),
    }
    RESULT.write_text(json.dumps(artifact, indent=2))
    _log(f"WROTE {RESULT.name}: solved={len(solved)}/{len(rows)} "
         f"med_noop={summary['median_noop_frac_solved']} "
         f"med_revisit={summary['median_revisit_frac_solved']} "
         f"med_disc1={summary['median_discovery_to_first_solve']} "
         f"med_actions={summary['median_total_actions_solved']}")
    print("VERDICT:", artifact["honest_verdict"], flush=True)


if __name__ == "__main__":
    main()

"""The FOCUSED ARC-AGI-3 LOOP — the measurement engine for rapid leaderboard progress.

Scores the CarnotAgent on the public games FROM SCRATCH (force-explore: the unseen-game
proxy, since the real eval is hidden games) using the LEADERBOARD METRIC: per game,
levels_completed and efficiency = sum over solved levels of min(baseline/agent_actions,1)^2.
Writes a leaderboard-style scorecard AND a per-game GAP LOG (which games fail + the
failure signature) so the next iteration targets the worst gap, not a random change.

The loop cadence (one turn each, gated):
  1. RUN this harness  -> levels + efficiency + gap log (no quota, offline).
  2. READ the worst gap (a game stuck at L0, or a solved game with terrible efficiency).
  3. IMPROVE one ingredient (salience tiers / frontier-distance nav / status-masking /
     E3 world-model induction for a deep game) -- a single, attributable change.
  4. RE-RUN; keep the change only if levels or efficiency strictly improved (regression
     gate: the previously-solved games must not regress).
  5. Append the closed/!closed gap to the log (never-prune) and repeat.

Usage: arc_leaderboard_eval.py [--budget N] [--mode explore|replay]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

from arcengine import GameAction
from carnot.agentic import arc_solver_kit as kit
from carnot.agentic.arc_competition_agent import (
    CLAIMED, CarnotAgentPolicy, E3AgentPolicy, load_cross_game_value_head, load_solutions, _level_of,
)

_VALUE_HEAD = None      # BRIDGE: the cross-game value head, loaded once for the explorer_vh policy
_PROPOSER = None        # E3: a stronger world-model proposer (a bigger GGUF), loaded once + reused
_PROPOSER_REPO = ""     # repo substr for the E3 proposer (e.g. "Qwen3.6-35B-A3B"); "" = E3 default 12B
_VALUE_WEIGHT = 0.0     # A* blend weight for explorer_vh; 0 = depth-primary tiebreak (neutral, safe).
                        # weight 5 REGRESSED the live explorer (6/32 vs 8/32) -- the value head misroutes
                        # the depth-first-ride structure. Set via --value-weight for sweeps.


def _oracle_levels() -> dict:
    """The per-game levels our OFFLINE oracle (GameAdapter/OfflineSolver path) reproduces, read from
    ops/arc_solve_registry.yaml. This is the upper bound the FRAME-ONLY LIVE path is measured against:
    the honest 'live gap' = oracle_levels - live_frame_only_levels, per game and in total."""
    import yaml
    d = yaml.safe_load((REPO / "ops" / "arc_solve_registry.yaml").read_text())
    return {g["game"]: int(g.get("levels_reproduced", 0)) for g in d.get("games", [])
            if g.get("reproducibility") == "reproduced" and int(g.get("levels_reproduced", 0)) > 0}


def _build_policy(kind: str, game: str):
    """The LIVE agent policy under test -- NO banked solution, NO GameAdapter, NO internal-state reads
    (the unseen-game simulation). 'explorer' = tier-1 graph_explore only; 'e3' = the FULL competition
    cascade (graph_explore -> E3 executable-world-model induction on stall) that make_carnot_agent runs."""
    if kind == "e3":
        # INDUCTION ARM: optionally use a STRONGER world-model proposer (a bigger GGUF) -- the lever
        # where gemma-4-12B closed 0/6. One proposer instance is loaded + reused across games.
        global _PROPOSER
        if _PROPOSER_REPO and _PROPOSER is None:
            from carnot.agentic.arc_executable_world_model import LocalGGUFProposer
            # port 8920 (NOT the conductor's E3 default 8919): LocalGGUFProposer reuses any server on its
            # port WITHOUT a model check, so a shared port silently runs the wrong model (a stale gemma
            # server made an earlier "Qwen" run actually use gemma-12B). A distinct port isolates this
            # test from the conductor's E3 work and guarantees the requested model loads.
            _PROPOSER = LocalGGUFProposer(repo_substr=_PROPOSER_REPO, port=8920)
        return E3AgentPolicy(game, proposer=_PROPOSER)
    if kind in ("explorer_vh", "explorer_bf"):              # BRIDGE: value-head-routed explorer
        global _VALUE_HEAD
        if _VALUE_HEAD is None:
            _VALUE_HEAD = load_cross_game_value_head()
        # explorer_bf = BEST-FIRST search (the graph_explore form where the value head's routing helped,
        # unlocking cn04); explorer_vh = the default depth-first-ride with an A*-frontier nudge.
        mode = "best_first" if kind == "explorer_bf" else "depth_first_ride"
        return CarnotAgentPolicy(game, {}, force_explore=True, value_head=_VALUE_HEAD,
                                 value_weight=_VALUE_WEIGHT, search_mode=mode)
    return CarnotAgentPolicy(game, {}, force_explore=True)   # force_explore -> ignores any banked plan


def _baseline_actions(env, game: str) -> dict:
    """Per-level human/reference action counts if the env exposes them (efficiency
    denominator). Best-effort; returns {} if unavailable offline."""
    for attr in ("baseline_actions", "human_actions", "reference_actions"):
        v = getattr(getattr(env, "info", env), attr, None)
        if v:
            return {i: int(x) for i, x in enumerate(v)} if isinstance(v, (list, tuple)) else dict(v)
    return {}


def run_game(game: str, policy, *, budget: int, variant: int = 0, reflect=None) -> dict:
    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    if variant:
        # MANUFACTURED held-out layout variant (operator 2026-06-19): the agent observes a mechanic-
        # preserving recolor/reflection it has never seen, while the REAL env keeps the win logic -> a
        # solve here is a real solve and a genuine generic-transfer test (a bigger benchmark than the
        # 2/7 LOO on 25 games). See arc_variant_generator.VariantEnv.
        from carnot.agentic.arc_variant_generator import VariantEnv

        env = VariantEnv(env, game, variant, reflect=reflect)
    base = _baseline_actions(env, game)
    frames, latest, actions = [], None, 0
    start = None
    best = None
    level_up_actions: list[int] = []  # cumulative `actions` count at each level-up (for per-level cost)
    for _ in range(budget):
        if policy.is_done(frames, latest):
            break
        kind, data = policy.next_move(frames, latest)
        if kind == "RESET":
            latest = env.reset()
        elif kind is None:
            break
        else:
            latest = env.step(getattr(GameAction, f"ACTION{kind}"), data=data)
            actions += 1
        if start is None:
            start = _level_of(latest)
            best = start
        lvl = _level_of(latest)
        if best is not None and lvl > best:
            for _lv in range(best, lvl):  # record the action count for each new level (handles jumps)
                level_up_actions.append(actions)
            best = lvl
        frames.append(latest)
        if latest is None:
            break
    reached = _level_of(latest)
    levels = max(0, reached - (start or 0))
    # REAL leaderboard efficiency is PER-LEVEL, not whole-game (docs/research-notes/
    # arc-agi3-kaggle-submission-requirements-2026-06-17.md: "per-level score =
    # min(human_actions/agent_actions, 1.0), squared"). agent_actions for a level = actions spent BETWEEN
    # consecutive level-ups, NOT the total run. The OLD formula `levels * min(baseline[0]/TOTAL_actions,1)^2`
    # was wrong: it scored an efficient-but-over-running solve at ~0 (lp85 solved L1 in 20 actions but ran
    # to 7792 hunting unreachable deeper levels -> old eff ~0). Under the correct per-level metric L1 scores
    # min(17/20,1)^2 = 0.72 and the over-run does NOT touch it (the over-run is a WALL-CLOCK cost, not a
    # score cost). This is the metric the local submission gate must judge configs on.
    per_level = []
    per_level_eff = 0.0
    prev = 0
    for k, lu in enumerate(level_up_actions):
        lvl_actions = lu - prev
        prev = lu
        human = base.get((start or 0) + k)
        ratio = min(human / lvl_actions, 1.0) if (human and lvl_actions > 0) else 1.0
        term = round(ratio * ratio, 4)
        per_level_eff += term
        per_level.append({"level": (start or 0) + k, "agent_actions": lvl_actions,
                          "human_actions": human, "efficiency": term})
    eff = round(per_level_eff, 4)
    gap = None
    if reached <= (start or 0):
        gap = {"game": game, "stuck_at_level": reached, "actions_spent": actions,
               "signature": "no_level_up_within_budget",
               "needs": "richer exploration (salience tiers / frontier-dist nav) OR E3 world-model induction"}
    return {"game": game, "levels": levels, "reached": reached, "actions": actions,
            "efficiency": eff, "per_level": per_level,
            "actions_to_first_levelup": (level_up_actions[0] if level_up_actions else None),
            "gap": gap}


def _arg(argv, flag, default):
    return argv[argv.index(flag) + 1] if flag in argv else default


def main() -> int:
    argv = sys.argv[1:]
    # --games oracle: measure the LIVE frame-only agent against the 16 games our OFFLINE oracle solved,
    #   reporting the honest GAP (oracle 32 levels - what the live path reaches with NO per-game knowledge).
    #   This is the north-star metric: live capability, not the offline reproducibility scorecard.
    # --policy e3: the FULL competition cascade (graph_explore + E3 world-model induction). default
    #   'explorer' = the fast tier-1 graph_explore floor (no LLM); 'e3' needs the local GGUF + is slower.
    games_mode = _arg(argv, "--games", "claimed")
    policy_kind = _arg(argv, "--policy", "explorer")
    budget = int(_arg(argv, "--budget", "20000"))
    global _VALUE_WEIGHT, _PROPOSER_REPO
    _VALUE_WEIGHT = float(_arg(argv, "--value-weight", "0"))   # explorer_vh A* blend weight (0 = neutral)
    _PROPOSER_REPO = _arg(argv, "--proposer", "")             # e3 stronger proposer repo substr ("" = 12B)
    oracle = _oracle_levels()
    games = sorted(oracle) if games_mode == "oracle" else list(CLAIMED)
    only = _arg(argv, "--only", "")          # --only g1,g2 : target a subset (e.g. the worst gaps)
    variant = int(_arg(argv, "--variant", "0"))  # 0 = real game; N>0 = manufactured held-out variant N
    reflect_arg = _arg(argv, "--reflect", "")     # "" none; "0"/"1" = reflect axis (vertical/horizontal)
    reflect = int(reflect_arg) if reflect_arg != "" else None
    if only:
        keep = set(only.split(","))
        games = [g for g in games if g in keep]
    print(f"== ARC LIVE-loop eval — games={games_mode} policy={policy_kind} budget={budget} "
          f"(frame-only, no banked plan, no GameAdapter) ==", flush=True)
    rows, total_levels, total_eff, gaps = [], 0, 0.0, []
    live_levels_sum, oracle_sum, gap_sum = 0, 0, 0
    for game in games:
        t0 = time.time()
        r = run_game(game, _build_policy(policy_kind, game), budget=budget, variant=variant, reflect=reflect)
        if games_mode == "oracle":
            r["oracle_levels"] = oracle.get(game, 0)
            r["gap_vs_oracle"] = max(0, oracle.get(game, 0) - r["levels"])
            live_levels_sum += r["levels"]
            oracle_sum += r["oracle_levels"]
            gap_sum += r["gap_vs_oracle"]
        rows.append(r)
        total_levels += r["levels"]
        total_eff += r["efficiency"]
        if r["gap"]:
            gaps.append(r["gap"])
        extra = (f" vs oracle L{r['oracle_levels']} (gap {r['gap_vs_oracle']})" if games_mode == "oracle" else "")
        print(f"  {game:5} live=L{r['reached']} (+{r['levels']}) actions={r['actions']:5} "
              f"eff={r['efficiency']:.4f}{extra}  [{time.time()-t0:.0f}s]", flush=True)
    if games_mode == "oracle":
        print(f"\n  LIVE-vs-ORACLE GAP: frame-only live path reaches {live_levels_sum}/{oracle_sum} "
              f"oracle levels (gap {gap_sum}). Closed: "
              f"{sorted(g for g in games if oracle.get(g, 0) and rows[games.index(g)]['levels'] >= oracle[g])}", flush=True)
    else:
        print(f"\n  LEADERBOARD SCORE: {total_levels} levels, efficiency-sum {total_eff:.3f}; "
              f"{len(gaps)} open gaps", flush=True)
    verdict = (f"complete_live_oracle_gap_{live_levels_sum}_of_{oracle_sum}_levels_gap_{gap_sum}"
               if games_mode == "oracle" else
               f"complete_leaderboard_eval_{total_levels}_levels_{len(gaps)}_gaps")
    out = REPO / "results" / ("arc_live_oracle_gap.json" if games_mode == "oracle" else "arc_leaderboard_eval.json")
    out.write_text(json.dumps({
        "experiment": "arc_live_oracle_gap" if games_mode == "oracle" else "arc_leaderboard_eval",
        "games_mode": games_mode, "policy": policy_kind, "budget": budget,
        "live_levels": live_levels_sum if games_mode == "oracle" else total_levels,
        "oracle_levels": oracle_sum, "gap": gap_sum,
        "efficiency_sum": round(total_eff, 4), "open_gaps": gaps, "per_game": rows,
        "inference_substrate": "offline_sim_no_quota_frame_only_live_agent",
        "honest_verdict": verdict,
    }, indent=2))
    print(f"  wrote {out.relative_to(REPO)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

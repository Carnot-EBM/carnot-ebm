#!/usr/bin/env python3
"""#2 Energy-as-fitness quality-diversity (QD) evolution over action-sequences.

THE BET (2026-06-22, .425 energy-config-space lead): turn the verifier into a
NON-AUTOREGRESSIVE GENERATOR. Energy provably fails to generate de-novo (P0.1), but
RANKING recombinations of REAL exploration fragments is a different, easier job.
MAP-Elites maintains a diverse archive of multi-action genomes seeded from real
exploration; the key operator is CROSSOVER AT A SHARED VISITED-STATE HASH — splice
genome A's reach-prefix (A reaches state s) onto genome B's continuation from s, so a
reach-prefix and a goal-suffix discovered SEPARATELY compose into a coherent plan no
single AR rollout or single-action BFS expansion produces. Fitness is a DENSE energy
along the rollout; the winner is a genome that banks a NEW level, offline-reproduced.

PERCEPTION-INDEPENDENT BY DESIGN: unlike #1, the fitness uses NO per-game goal
featurizer (level progress + state-coverage + action-effect density), so it sidesteps
the per-game-perception gate that bounds #1 (the Tier-A finding). The energy ensemble
(frame-change scorer / cell-recall / a learned verifier) plugs into `genome_fitness`
as a richer dense signal later — the default here is the perception-free baseline.

Compared at MATCHED total-env-step budget against blind BFS (graph_explore_solve_v2):
does QD bank a reproduced winner BFS does NOT? Gate (research note): >=2 of the hard
games. Honest, reproduction-gated, OFFLINE, zero quota. verifier_is_oracle: false.
OUTER-LOOP PREP EXPERIMENT (not a conductor task) — run AFTER the conductor is stopped.

  .venv/bin/python scripts/experiments/experiment_2_energy_as_fitness_qd.py \
      --games ls20,tu93,wa30 --budget 8000 --seeds 40
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
from arcengine import GameAction
from carnot.agentic import arc_solver_kit as kit
from carnot.agentic.arc_agi3_world_model import grid_of, frame_hash
from carnot.agentic.arc_agi3_live_adapter import _game_action, _levels_completed
from carnot.agentic.arc_graph_explore import (
    graph_explore_solve_v2, trajectory_labels, rich_action_candidates, _warm,
)

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "results" / "experiment_2_energy_as_fitness_qd.json"


# ---- action-label genome plumbing (the arc_gap_fill label format) ----
def _label(action_id: int, data) -> str:
    return json.dumps({"action": int(action_id), "data": data})


def _apply(env, label: str, frame):
    s = json.loads(label)
    return env.step(_game_action(GameAction, s["action"]), data=s.get("data"))


def _level(frame) -> int:
    return int(_levels_completed(frame))


def _hash(frame):
    try:
        return frame_hash(np.asarray(grid_of(frame)))
    except Exception:
        return None


# ---- evaluation: replay a genome on a fresh deterministic offline env ----
def _new_env(game: str):
    arc = kit.offline_arcade()
    return arc.make(game, scorecard_id=arc.open_scorecard())


def rollout(game: str, genome: list[str]) -> dict:
    """Replay genome from reset; return level reached + the dense-fitness signals +
    the prefix-index -> state-hash map (for crossover splice points)."""
    env = _new_env(game)
    f = _warm(env, False)
    start = _level(f)
    reached = start
    prev_h = _hash(f)
    hashes = {0: prev_h}
    distinct = {prev_h}
    changed = 0
    for i, lab in enumerate(genome):
        nf = _apply(env, lab, f)
        if nf is None:
            break
        f = nf
        h = _hash(f)
        hashes[i + 1] = h
        if h is not None:
            if h != prev_h:
                changed += 1
            distinct.add(h)
            prev_h = h
        reached = max(reached, _level(f))
    return {
        "start_level": start, "reached_level": reached, "won": reached > start,
        "n_distinct": len(distinct), "frame_change_count": changed,
        "len": len(genome), "hashes": hashes,
    }


# ---- seeding: live exploration that collects REAL trajectories + an action vocab ----
def seed_rollout(game: str, max_len: int, rng: random.Random) -> dict:
    """One live exploration rollout: at each step pick a random salient candidate
    (real rich_action_candidates), recording the genome, its visited hashes, and the
    candidate labels seen (the action vocabulary that mutation will sample from)."""
    env = _new_env(game)
    f = _warm(env, False)
    start = _level(f)
    reached = start
    genome: list[str] = []
    vocab: set[str] = set()
    prev_h = _hash(f)
    hashes = {0: prev_h}
    distinct = {prev_h}
    changed = 0
    for i in range(max_len):
        cands = rich_action_candidates(f) if f is not None else []
        if not cands:
            break
        for c in cands[:12]:
            vocab.add(_label(int(c.action_id), c.data))
        c = cands[rng.randrange(min(len(cands), 8))]
        lab = _label(int(c.action_id), c.data)
        nf = _apply(env, lab, f)
        if nf is None:
            break
        genome.append(lab)
        f = nf
        h = _hash(f)
        hashes[len(genome)] = h
        if h is not None:
            if h != prev_h:
                changed += 1
            distinct.add(h)
            prev_h = h
        reached = max(reached, _level(f))
        if reached > start:
            break
    return {
        "genome": genome, "vocab": vocab, "start_level": start, "reached_level": reached,
        "won": reached > start, "n_distinct": len(distinct),
        "frame_change_count": changed, "len": len(genome), "hashes": hashes,
    }


# ---- energy-as-fitness (perception-independent dense signal) ----
def genome_fitness(r: dict) -> float:
    """Dense progress energy: level progress dominates; state-coverage + action-effect
    density reward genomes that reach novel/controllable states even before a level-up;
    a small length penalty prefers efficient plans. (Plug a learned energy verifier /
    frame-change scorer here for a richer dense signal — this is the perception-free
    default.)"""
    return (1000.0 * (r["reached_level"] - r["start_level"])
            + float(r["n_distinct"]) + 0.5 * float(r["frame_change_count"])
            - 0.1 * float(r["len"]))


def descriptor(r: dict) -> tuple:
    """MAP-Elites behavior descriptor (perception-free): keeps diversity across reached
    level, plan length, and exploration coverage so the archive doesn't collapse onto
    one near-miss (the diversity-floor failure mode)."""
    return (int(r["reached_level"]), int(r["len"]) // 5, int(r["n_distinct"]) // 8)


# ---- variation operators ----
def mutate(genome: list[str], vocab: list[str], rng: random.Random) -> list[str]:
    g = list(genome)
    if not g:
        return [vocab[rng.randrange(len(vocab))]] if vocab else g
    op = rng.randrange(5)
    if op == 0 and vocab:  # insert
        g.insert(rng.randrange(len(g) + 1), vocab[rng.randrange(len(vocab))])
    elif op == 1:  # delete
        g.pop(rng.randrange(len(g)))
    elif op == 2 and len(g) >= 2:  # swap
        i, j = rng.randrange(len(g)), rng.randrange(len(g))
        g[i], g[j] = g[j], g[i]
    elif op == 3:  # truncate
        g = g[: rng.randrange(1, len(g) + 1)]
    elif vocab:  # extend with a few salient actions
        for _ in range(rng.randint(1, 3)):
            g.append(vocab[rng.randrange(len(vocab))])
    return g


def crossover(ga: list[str], ha: dict, gb: list[str], hb: dict, rng: random.Random):
    """THE non-AR operator: splice A's reach-prefix onto B's continuation at a SHARED
    visited-state hash. A reaches state s at prefix index i; B is at s at index j;
    child = A[:i] + B[j:] reaches s via A then continues via B — composing two
    separately-discovered fragments. Returns None if A and B share no visited state."""
    a_by_hash = {h: i for i, h in ha.items() if h is not None}
    shared = [(a_by_hash[h], j) for j, h in hb.items() if h is not None and h in a_by_hash]
    shared = [(i, j) for (i, j) in shared if 0 < i <= len(ga) and 0 <= j <= len(gb)]
    if not shared:
        return None
    i, j = shared[rng.randrange(len(shared))]
    child = ga[:i] + gb[j:]
    return child if child and child != ga and child != gb else None


# ---- BFS baseline at matched budget ----
def bfs_baseline(game: str, max_expansions: int) -> dict:
    env = _new_env(game)
    st: dict = {}
    traj, lvl = graph_explore_solve_v2(env, 0, max_expansions=max_expansions, max_depth=60, stats=st)
    won = bool(traj) and int(lvl) >= 1
    reproduced = False
    if won:
        g = kit.reproduce(game, trajectory_labels(traj), _apply, claimed_level=int(lvl))
        reproduced = bool(g["reproduced"])
    return {"won": won, "reached_level": int(lvl), "offline_reproduced": reproduced,
            "expansions": st.get("expansions"), "actions": len(traj) if traj else 0}


# ---- the QD run for one game ----
def qd_solve(game: str, budget: int, n_seeds: int, seed_len: int, rng: random.Random) -> dict:
    archive: dict[tuple, dict] = {}   # descriptor -> {fitness, genome, rollout}
    vocab: set[str] = set()
    steps_used = 0
    best_won = None

    def consider(genome: list[str], r: dict):
        nonlocal best_won
        cell = descriptor(r)
        fit = genome_fitness(r)
        cur = archive.get(cell)
        if cur is None or fit > cur["fitness"]:
            archive[cell] = {"fitness": fit, "genome": genome, "rollout": r}
        if r["won"] and best_won is None:
            best_won = {"genome": genome, "reached_level": r["reached_level"], "len": r["len"]}

    # 1) seed from real exploration
    for _ in range(n_seeds):
        if steps_used >= budget:
            break
        s = seed_rollout(game, seed_len, rng)
        steps_used += max(1, s["len"])
        vocab |= s["vocab"]
        consider(s["genome"], {k: s[k] for k in
                               ("start_level", "reached_level", "won", "n_distinct",
                                "frame_change_count", "len", "hashes")})
    vocab_l = sorted(vocab)
    generations = 0

    # 2) evolve: mutation + crossover, MAP-Elites insertion, until budget
    while steps_used < budget and archive and best_won is None:
        generations += 1
        cells = list(archive.values())
        if rng.random() < 0.5 and len(cells) >= 2:   # crossover (the non-AR operator)
            a, b = rng.sample(cells, 2)
            child = crossover(a["genome"], a["rollout"]["hashes"],
                              b["genome"], b["rollout"]["hashes"], rng)
        else:                                          # mutation
            p = cells[rng.randrange(len(cells))]
            child = mutate(p["genome"], vocab_l, rng)
        if not child:
            continue
        r = rollout(game, child)
        steps_used += max(1, r["len"])
        consider(child, r)

    return {
        "game": game, "steps_used": steps_used, "generations": generations,
        "archive_size": len(archive), "vocab_size": len(vocab_l),
        "best_won": best_won, "seeded_won": any(c["rollout"]["won"] for c in archive.values()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=str, default="ls20,tu93,wa30")
    ap.add_argument("--budget", type=int, default=8000, help="total env steps per game (QD and BFS matched)")
    ap.add_argument("--seeds", type=int, default=40)
    ap.add_argument("--seed-len", type=int, default=40)
    ap.add_argument("--seed", type=int, default=20260622)
    args = ap.parse_args()
    t0 = time.time()
    games = [g.strip() for g in args.games.split(",") if g.strip()]

    rows = []
    for game in games:
        rng = random.Random(args.seed + hash(game) % 9999)
        t1 = time.time()
        qd = qd_solve(game, args.budget, args.seeds, args.seed_len, rng)
        # match BFS budget to QD's total env steps (expansions ~ env steps)
        bfs = bfs_baseline(game, max_expansions=max(qd["steps_used"], args.budget))
        qd_won = qd["best_won"] is not None
        # reproduction-gate the QD winner
        qd_repro = False
        if qd_won:
            g = kit.reproduce(game, qd["best_won"]["genome"], _apply,
                              claimed_level=int(qd["best_won"]["reached_level"]))
            qd_repro = bool(g["reproduced"])
        row = {
            "game": game,
            "qd_won": qd_won, "qd_offline_reproduced": qd_repro,
            "qd_reached_level": qd["best_won"]["reached_level"] if qd_won else None,
            "qd_winner_len": qd["best_won"]["len"] if qd_won else None,
            "bfs_won": bfs["won"], "bfs_offline_reproduced": bfs["offline_reproduced"],
            "bfs_reached_level": bfs["reached_level"],
            "qd_generates_where_bfs_does_not": bool(qd_repro and not bfs["offline_reproduced"]),
            "steps_used": qd["steps_used"], "generations": qd["generations"],
            "archive_size": qd["archive_size"], "vocab_size": qd["vocab_size"],
            "secs": round(time.time() - t1, 1),
        }
        rows.append(row)
        print(f"  [{game}] qd_won={qd_won} repro={qd_repro} (L{row['qd_reached_level']}) "
              f"| bfs_won={bfs['won']} repro={bfs['offline_reproduced']} (L{bfs['reached_level']}) "
              f"| QD>BFS={row['qd_generates_where_bfs_does_not']} "
              f"gens={qd['generations']} arch={qd['archive_size']} [{row['secs']}s]", flush=True)

    n_qd_only = sum(1 for r in rows if r["qd_generates_where_bfs_does_not"])
    n_qd_repro = sum(1 for r in rows if r["qd_offline_reproduced"])
    if n_qd_only >= 2:
        verdict = "success: energy_as_fitness_qd_generates_above_bfs_on_2plus_games"
    elif n_qd_only >= 1:
        verdict = "complete: energy_as_fitness_qd_generates_above_bfs_on_1_game_preliminary"
    elif n_qd_repro >= 1:
        verdict = "complete: energy_as_fitness_qd_reproduces_but_not_above_bfs_honest_null_gap_sharpened"
    else:
        verdict = "complete: energy_as_fitness_qd_no_winner_honest_null_gap_sharpened"

    artifact = {
        "experiment": "experiment_2_energy_as_fitness_qd",
        "honest_verdict": verdict,
        "verifier_is_oracle": False,
        "inference_substrate": "offline_arc_search",
        "random_seed": args.seed,
        "budget_env_steps_per_game": args.budget,
        "games": games,
        "n_games_qd_generates_above_bfs": n_qd_only,
        "n_games_qd_reproduced": n_qd_repro,
        "gate_note": "research-note gate = QD banks a reproduced winner BFS does NOT on >=2 of the hard games",
        "fitness_note": ("perception-independent: 1000*level_progress + n_distinct_states + "
                         "0.5*frame_change_count - 0.1*len; energy verifier plugs into genome_fitness"),
        "non_ar_operator": "crossover at a shared visited-state hash (reach-prefix + goal-suffix splice)",
        "rows": rows,
        "duration_s": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(artifact, indent=2))
    print(f"\nVERDICT: {verdict}")
    print(f"  QD-generates-above-BFS on {n_qd_only}/{len(games)} games; QD reproduced on {n_qd_repro}. -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

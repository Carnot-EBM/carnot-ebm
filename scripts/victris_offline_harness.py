"""Offline Victris Arena harness: measure a bot's win rate in the local simulator only.

Never connects to the live ladder. Needs the `arena-sim` package (pip install arena-sim).
Bots are `decide(obs, hello) -> turn` where turn is the ladder's JSON turn without `tick`.
Seeds are split: dev seeds for iterating, held-out seeds only for the final number.
Usage: python -I scripts/victris_offline_harness.py --bot module:func [--opponents ts-starter random]
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import random
from collections.abc import Callable

from arena_sim import ArenaEnv

Bot = Callable[[dict, dict], dict]
DEV_SEEDS = range(0, 1000)
HELDOUT_SEEDS = range(10_000, 11_000)  # never tune against these


def shoot_bot(obs: dict, hello: dict) -> dict:
    """Baseline: step toward the enemy and shoot when it is visible."""
    turn: dict = {}
    if obs["enemies"]:
        e = obs["enemies"][0]["pos"]
        if obs["moves"]:  # empty while stunned or rooted
            best = min(
                obs["moves"], key=lambda m: abs(m["tile"][0] - e[0]) + abs(m["tile"][1] - e[1])
            )
            turn["move"] = best["dir"]
        if obs["self"]["cooldowns"]["ranged"] == 0:
            turn["attack"] = {"name": "ranged", "target": e}
    elif obs["moves"]:
        z = obs["zone"]["center"]
        turn["move"] = min(
            obs["moves"], key=lambda m: abs(m["tile"][0] - z[0]) + abs(m["tile"][1] - z[1])
        )["dir"]
    return turn


def idle_bot(obs: dict, hello: dict) -> dict:
    return {}


def play(bot: Bot, opponent: str, seed: int) -> dict:
    env = ArenaEnv(opponent)
    obs, _ = env.reset(seed=seed)
    hello = env._hello
    while True:
        turn = {**bot(obs, hello), "tick": obs["tick"]}
        nxt, _, term, trunc, info = env.step(turn)
        if term or trunc:
            env.close()
            return info.get("end", {})
        obs = nxt


def wilson(w: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = w / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def evaluate(bot: Bot, opponent: str, seeds, progress: bool = True) -> dict:
    wins = draws = 0
    n = 0
    for s in seeds:
        end = play(bot, opponent, s)
        r = end.get("result")
        wins += r == "win"
        draws += r == "draw"
        n += 1
        if progress and n % 25 == 0:
            print(f"progress {opponent} {n} wins={wins}", flush=True)
    lo, hi = wilson(wins, n)
    return {
        "opponent": opponent,
        "n": n,
        "wins": wins,
        "draws": draws,
        "win_rate": wins / max(n, 1),
        "ci95": [lo, hi],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bot", default="shoot")
    ap.add_argument("--opponents", nargs="+", default=["ts-starter", "random"])
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--heldout", action="store_true", help="use held-out seeds (final number only)")
    a = ap.parse_args()
    if a.bot in ("shoot", "idle"):
        bot = {"shoot": shoot_bot, "idle": idle_bot}[a.bot]
    else:
        mod, fn = a.bot.split(":")
        bot = getattr(importlib.import_module(mod), fn)
    pool = list(HELDOUT_SEEDS if a.heldout else DEV_SEEDS)
    random.Random(0).shuffle(pool)
    out = [evaluate(bot, o, pool[: a.n]) for o in a.opponents]
    print(json.dumps({"split": "heldout" if a.heldout else "dev", "results": out}, indent=1))


if __name__ == "__main__":
    main()

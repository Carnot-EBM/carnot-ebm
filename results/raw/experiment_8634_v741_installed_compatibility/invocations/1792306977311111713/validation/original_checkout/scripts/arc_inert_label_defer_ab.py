#!/usr/bin/env python3
"""REQ-ARC-WMTE-6071 A/B: does deferring already-known-inert action labels buy PROGRESS?

THREE ARMS, and the third is the one that makes the other two readable:

  * ``control``   -- the shipped configuration (``CARNOT_ARC_INERT_LABEL_DEFER`` unset).
  * ``control_b`` -- byte-identical configuration to ``control``, in a SEPARATE process. This is
    the A/A noise floor. If every ``control``/``control_b`` pair hashes to the same action trace
    then the agent is DETERMINISTIC at a fixed seed, and any ``defer`` divergence is causal
    rather than sampled -- a far stronger statement than a p-value over 25 games.
  * ``defer``     -- identical except ``CARNOT_ARC_INERT_LABEL_DEFER=1``.

THE TRAP THIS DESIGN EXISTS TO AVOID. An "efficient" explorer that simply EXPLORES LESS also
spends fewer actions, so raw action count is not the metric. Both arms are given the SAME action
budget and the comparison is on what they REACHED with it: levels (the oracle), per-level
hand-verifier progress, and distinct states discovered. An arm that saves actions and reaches
less is a REGRESSION and is reported as one.

STATISTICS, stated BEFORE the run. Replicates of one game are not independent trials -- the
seeds share a game's entire mechanics -- so the unit of analysis is the GAME (25 units), with
seeds averaged within a game first. The test is a two-sided paired sign test over games whose
paired difference is non-zero. That makes the MINIMUM REACHABLE p a function of how many games
move at all: with k discordant games, min p = 2 * 0.5**k, so p <= 0.05 needs k >= 6 and p <= 0.01
needs k >= 8. If fewer than 6 games move, THE DESIGN CANNOT REACH 0.05 and the result is reported
as a descriptive effect with that limit stated, not as a null.

A crash or a timeout is a MISSING OBSERVATION. Cells that error are excluded and counted, never
folded in as a zero.

Never plays a scored or online game; never starts a generator; never touches a GPU.

Usage:
    python scripts/arc_inert_label_defer_ab.py --out-dir results/arc_inert_label_defer_20260802
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

REPO = Path(__file__).resolve().parent.parent
PY = sys.executable

ARM_ENV = {
    "control": {},
    "control_b": {},
    "defer": {"CARNOT_ARC_INERT_LABEL_DEFER": "1"},
}


def score(baseline_actions: float, agent_actions: float) -> float:
    """ARC-AGI-3's per-level score: ``min((baseline/agent)**2, 115)``."""

    if agent_actions <= 0:
        return 115.0
    return min((float(baseline_actions) / float(agent_actions)) ** 2, 115.0)


def run_cell(game: str, arm: str, seed: int, out_dir: Path, budget: int, scratch: Path) -> Path:
    out = out_dir / "cells" / f"{arm}__{game}__{seed}.json"
    if out.exists():
        return out
    env = dict(os.environ)
    # PER-ARM scratch engine store. One shared store would let arm A's induced engine sit on
    # disk when arm B starts, so the comparison would be between two different situations while
    # being reported as one situation with and without a lever.
    e3 = scratch / "e3" / arm / f"{game}_{seed}"
    e3.mkdir(parents=True, exist_ok=True)
    env["CARNOT_ARC_E3_DIR"] = str(e3)
    env["CARNOT_ARC_DISABLE_INDUCTION"] = "1"
    # An empty visible-device list makes "this run touched no GPU" structural, not a promise.
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["PYTHONPATH"] = str(REPO / "python")
    env.pop("CARNOT_ARC_INERT_LABEL_DEFER", None)
    env.update(ARM_ENV[arm])
    cmd = [
        PY,
        str(REPO / "scripts" / "arc_inert_label_defer_worker.py"),
        "--game",
        game,
        "--seed",
        str(seed),
        "--budget",
        str(budget),
        "--arm",
        arm,
        "--out",
        str(out),
    ]
    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=2400)
        rc, tail = proc.returncode, proc.stderr[-2000:]
    except subprocess.TimeoutExpired:
        rc, tail = -9, "timeout"
    if rc != 0:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {"game": game, "arm": arm, "seed": seed, "error": f"exit {rc}", "stderr": tail},
                indent=1,
            )
        )
    return out


def sign_test_p(wins: int, losses: int) -> Optional[float]:
    """Two-sided exact binomial sign test at p=0.5. None when nothing is discordant."""

    from math import comb

    n = wins + losses
    if n == 0:
        return None
    k = min(wins, losses)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2**n)
    return min(1.0, 2 * tail)


def _mean(vals: list[float]) -> Optional[float]:
    vals = [v for v in vals if v is not None]
    return statistics.fmean(vals) if vals else None


def analyse(cells: dict[str, dict[tuple[str, int], dict]], games, seeds) -> dict[str, Any]:
    """Game-clustered paired analysis. Seeds are averaged WITHIN a game first."""

    per_game: dict[str, Any] = {}
    for g in games:
        row: dict[str, Any] = {"seeds": []}
        for arm in ("control", "control_b", "defer"):
            got = [cells[arm].get((g, s)) for s in seeds]
            ok = [c for c in got if c and not c.get("error")]
            row[f"{arm}_n_ok"] = len(ok)
            row[f"{arm}_levels"] = _mean([c["levels_gained"] for c in ok])
            row[f"{arm}_actions"] = _mean([c["actions_spent"] for c in ok])
            row[f"{arm}_states"] = _mean([c["states_discovered"] for c in ok])
            row[f"{arm}_inert"] = _mean([c["inert_actions"] for c in ok])
            row[f"{arm}_nav"] = _mean([c["navigation_actions"] for c in ok])
            row[f"{arm}_hv"] = _mean([c.get("hv_progress_best_level") for c in ok])
            a2f = [c["actions_to_first_levelup"] for c in ok if c["actions_to_first_levelup"]]
            row[f"{arm}_a2f"] = _mean(a2f) if len(a2f) == len(ok) and ok else None
            row[f"{arm}_traces"] = [c.get("trace_sha256") for c in ok]
        # A/A: did the two identically-configured arms produce the same trace on every seed?
        aa = [
            (cells["control"].get((g, s)) or {}).get("trace_sha256")
            == (cells["control_b"].get((g, s)) or {}).get("trace_sha256")
            for s in seeds
            if cells["control"].get((g, s)) and cells["control_b"].get((g, s))
        ]
        row["aa_identical_all_seeds"] = bool(aa) and all(aa)
        row["defer_diverged_from_control"] = any(
            (cells["control"].get((g, s)) or {}).get("trace_sha256")
            != (cells["defer"].get((g, s)) or {}).get("trace_sha256")
            for s in seeds
            if cells["control"].get((g, s)) and cells["defer"].get((g, s))
        )
        row["defer_fired"] = sum(
            int(
                ((cells["defer"].get((g, s)) or {}).get("inert_label_defer_diagnostics") or {}).get(
                    "deferred_pops", 0
                )
            )
            for s in seeds
        )
        per_game[g] = row

    def paired(metric: str, higher_is_better: bool) -> dict[str, Any]:
        wins = losses = ties = 0
        deltas = []
        movers = []
        for g in games:
            c, d = per_game[g].get(f"control_{metric}"), per_game[g].get(f"defer_{metric}")
            if c is None or d is None:
                continue
            delta = d - c
            deltas.append(delta)
            if delta == 0:
                ties += 1
                continue
            good = delta > 0 if higher_is_better else delta < 0
            movers.append({"game": g, "control": c, "defer": d, "delta": round(delta, 4)})
            if good:
                wins += 1
            else:
                losses += 1
        k = wins + losses
        return {
            "metric": metric,
            "higher_is_better": higher_is_better,
            "games_scored": len(deltas),
            "games_better": wins,
            "games_worse": losses,
            "games_tied": ties,
            "pooled_control": round(sum(per_game[g][f"control_{metric}"] or 0 for g in games), 4),
            "pooled_defer": round(sum(per_game[g][f"defer_{metric}"] or 0 for g in games), 4),
            "mean_delta": round(statistics.fmean(deltas), 4) if deltas else None,
            "sign_test_p": sign_test_p(wins, losses),
            "min_reachable_p_given_k_discordant": (2 * 0.5**k) if k else None,
            "movers": sorted(movers, key=lambda r: r["delta"]),
        }

    return {
        "per_game": per_game,
        "paired": {
            "levels": paired("levels", True),
            "states": paired("states", True),
            "hv": paired("hv", True),
            "inert": paired("inert", False),
            "nav": paired("nav", False),
            "actions": paired("actions", False),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--budget", type=int, default=2000)
    ap.add_argument("--seeds", default="20260802,20260803,20260804")
    ap.add_argument("--games", default="")
    ap.add_argument("--arms", default="control,control_b,defer")
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--scratch", default="")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / "cells").mkdir(parents=True, exist_ok=True)
    scratch = Path(args.scratch or (out_dir / "_scratch"))
    scratch.mkdir(parents=True, exist_ok=True)

    if args.games:
        games = [g.strip() for g in args.games.split(",") if g.strip()]
    else:
        import yaml

        reg = yaml.safe_load((REPO / "ops" / "arc_solve_registry.yaml").read_text())
        games = sorted({g.get("game") for g in reg["games"] if g.get("game")})
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    t0 = time.time()
    jobs = [(g, a, s) for a in arms for g in games for s in seeds]
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        list(
            pool.map(
                lambda j: run_cell(j[0], j[1], j[2], out_dir, args.budget, scratch),
                jobs,
            )
        )

    cells: dict[str, dict[tuple[str, int], dict]] = {a: {} for a in arms}
    errors = []
    for a in arms:
        for g in games:
            for s in seeds:
                p = out_dir / "cells" / f"{a}__{g}__{s}.json"
                try:
                    c = json.loads(p.read_text())
                except Exception as exc:
                    c = {"error": f"unreadable:{exc}"}
                if c.get("error"):
                    errors.append({"arm": a, "game": g, "seed": s, "error": c["error"]})
                cells[a][(g, s)] = c

    result = analyse(cells, games, seeds)
    payload = {
        "games": games,
        "seeds": seeds,
        "arms": arms,
        "budget": args.budget,
        "duration_s": round(time.time() - t0, 3),
        "missing_observations": errors,
        "n_missing": len(errors),
        **result,
        "cells": {a: {f"{g}|{s}": cells[a][(g, s)] for g in games for s in seeds} for a in arms},
    }
    raw = json.dumps(payload, sort_keys=True, default=str).encode()
    payload["reproducibility_checksum"] = "sha256:" + hashlib.sha256(raw).hexdigest()
    (out_dir / "ab.json").write_text(json.dumps(payload, indent=1, default=str))
    print(json.dumps(payload["paired"], indent=1, default=str)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

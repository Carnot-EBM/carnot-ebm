"""LOCAL SUBMISSION GATE (operator directive 2026-06-20): never waste a 1/day Kaggle submission slot on a
config that is a LOCAL REGRESSION.

Before any submit, this measures the CURRENT submitted-default agent (make_carnot_agent -> E3AgentPolicy,
via `arc_leaderboard_eval.py --policy e3`, frame-only) on a fixed game set and compares it to the best
VERIFIED baseline (`ops/arc-submission-baseline.json`) on the two things the leaderboard actually rewards:
  (1) solve-rate (solved games), and (2) ACTION EFFICIENCY (median actions on solved games -- the score
  is (human/agent)^2, so a "solve" that burns the whole action budget scores ~0).
PASS only if the current config is NON-INFERIOR on BOTH. This is what catches the regressions we already
hit: value_weight=5 (1/8 solved, slow) and the E3+v3 cascade (3 solved, ~7700 actions/solve vs bare BFS's
21 on lp85). It is a LOCAL (25-public-game) proxy, NOT a leaderboard predictor -- its only claim is
"don't submit a config locally WORSE than the last verified one."

Exit 0 = PASS (safe to submit), 1 = FAIL (regression -> refuse), 2 = could not measure.
CLI:  --check (default)            run the gate, print verdict, set exit code
      --update-baseline            overwrite the baseline with the CURRENT measurement (after a verified
                                   improvement + an actual successful submit)
      --policy e3|explorer         which policy to measure (default e3 = the submitted default)
      --budget N (8000)  --cap S (115)  --json
"""
import argparse
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import median

REPO = Path(__file__).resolve().parents[2]
BASELINE = REPO / "ops" / "arc-submission-baseline.json"
EVAL = REPO / "scripts" / "arc_leaderboard_eval.py"
# 4 reliably-solvable games (the bare-BFS solves) + 4 controls. Small so the gate runs in a couple minutes.
GATE_GAMES = ["lp85", "m0r0", "sp80", "vc33", "cd82", "ft09", "su15", "ls20"]
_LINE = re.compile(r"live=L(\d+)\s*\(\+(\d+)\)\s*actions=\s*(\d+)")
EFFICIENCY_SLACK = 1.10  # allow 10% worse median actions before calling it a regression


def _measure_game(game: str, policy: str, budget: int, cap: int) -> dict:
    import os

    cmd = [str(REPO / ".venv" / "bin" / "python"), str(EVAL), "--policy", policy,
           "--games", "oracle", "--only", game, "--budget", str(budget)]
    # Measure the SEARCH/efficiency of the tier-1 explorer cleanly: disable the LLM induction tier so the
    # gate doesn't pay the local llama-server spawn (irrelevant to a search regression; a one-time cost
    # under the real 12h eval). Production submission does NOT set this -> induction runs normally there.
    env = {**os.environ, "CARNOT_ARC_DISABLE_INDUCTION": "1"}
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=cap, cwd=str(REPO), env=env)
        m = None
        for ln in p.stdout.splitlines():
            if game in ln and "live=L" in ln:
                m = _LINE.search(ln)
        if not m:
            return {"game": game, "timed_out": False, "solved": False, "actions": None}
        levels, actions = int(m.group(2)), int(m.group(3))
        return {"game": game, "timed_out": False, "solved": levels >= 1, "actions": actions}
    except subprocess.TimeoutExpired:
        return {"game": game, "timed_out": True, "solved": False, "actions": None}


def measure(policy: str, budget: int, cap: int) -> dict:
    with ThreadPoolExecutor(max_workers=8) as ex:
        rows = list(ex.map(lambda g: _measure_game(g, policy, budget, cap), GATE_GAMES))
    solved = [r for r in rows if r["solved"]]
    acts = [r["actions"] for r in solved if r["actions"] is not None]
    return {
        "policy": policy, "games": GATE_GAMES, "per_game": rows,
        "solved_count": len(solved),
        "median_actions_on_solved": (median(acts) if acts else None),
        "total_actions_on_solved": (sum(acts) if acts else None),
        "timed_out_count": sum(1 for r in rows if r["timed_out"]),
        # CORE set-containment keys (2026-06-20): the verdict compares the SAME games across
        # baseline/treatment, so a knife-edge marginal solve flipping due to order-perturbation
        # noise (A1/A2 lesson: a 5%-recall prune that removes ~nothing still reshuffled the chaotic
        # ~7800-action trajectory and dropped m0r0) is NOT counted as a regression. The CORE is the
        # set of games the BASELINE solves; a lever must preserve every CORE solve (solve-rate
        # non-regression by set, not by count) and cut median actions ON THAT FIXED SET.
        "solved_games": sorted(r["game"] for r in solved),
        "actions_by_game": {r["game"]: r["actions"] for r in solved if r["actions"] is not None},
    }


def _verdict(cur: dict, base: dict) -> tuple[bool, str]:
    """CORE set-containment verdict (2026-06-20 redesign).

    The OLD verdict compared raw solved_COUNT, so a lever that merely reordered the chaotic
    near-budget search and flipped one knife-edge solve 4<->3 FAILed automatically regardless of
    merit (A1/A2 both died this way with their positive_control passing -> the metric, not the
    lever, was broken). The NEW verdict:
      * CORE := the games the BASELINE solves. A lever must preserve EVERY core solve (set
        containment, not count) -- this is the only relaxation that still FAILs a config that
        trades core solves for fringe ones (e.g. A2 swapping 3 core for 2 fringe).
      * median actions is measured on the FIXED CORE denominator (+inf for any core game the
        treatment failed to solve), so savings are credited on the same games, never gamed by
        dropping a hard one.
      * new solves OUTSIDE core are a reported BONUS, NEVER netted against a core loss.
    A legacy fallback (raw count) keeps the gate working against an old baseline JSON until the
    next `--update-baseline` persists the CORE keys.
    """
    BIG = 10 ** 9
    # Reconstruct baseline per-game actions from the new key, else from the legacy per_game rows.
    base_acts = base.get("actions_by_game") or {
        r["game"]: r["actions"] for r in base.get("per_game", [])
        if r.get("solved") and r.get("actions") is not None
    }
    core = set(base.get("solved_games") or base_acts.keys())
    if not core:
        # No baseline solves recorded at all -> fall back to the legacy count check.
        bs, cs = base.get("solved_count", 0), cur["solved_count"]
        return (cs >= bs and cs > 0,
                f"legacy count check: solved {cs} vs baseline {bs} (run --update-baseline for CORE)")
    cur_solved = set(cur.get("solved_games") or [])
    lost = sorted(core - cur_solved)
    if lost:
        return False, f"REGRESSION: lost CORE solves {lost} (core={sorted(core)})"
    cur_acts = cur.get("actions_by_game") or {}
    cm = median([cur_acts.get(g, BIG) for g in core])
    bm = median([base_acts.get(g, BIG) for g in core])
    if cm > bm * EFFICIENCY_SLACK:
        return False, (f"REGRESSION: CORE median actions {cm} > baseline {bm} x{EFFICIENCY_SLACK} "
                       f"(action efficiency is the scoring metric)")
    bonus = sorted(cur_solved - core)  # extra solves: reported, NEVER netted against a core loss
    tag = "IMPROVED" if (cm < bm or bonus) else "non-inferior"
    msg = f"PASS ({tag}): CORE {sorted(core)} median actions {cm} vs baseline {bm}"
    if bonus:
        msg += f"; BONUS solves {bonus}"
    return True, msg


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--update-baseline", action="store_true")
    ap.add_argument("--policy", default="e3")
    ap.add_argument("--budget", type=int, default=8000)
    ap.add_argument("--cap", type=int, default=115)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    print(f"[gate] measuring current submitted config (policy={a.policy}) on {len(GATE_GAMES)} games "
          f"(budget {a.budget}, {a.cap}s cap each)...", flush=True)
    cur = measure(a.policy, a.budget, a.cap)

    if a.update_baseline:
        BASELINE.write_text(json.dumps({**cur, "note": "verified baseline (update only after a real "
                                        "improvement + successful submit)"}, indent=2))
        print(f"[gate] baseline UPDATED: solved {cur['solved_count']}, "
              f"median actions/solve {cur['median_actions_on_solved']}")
        return 0

    if not BASELINE.exists():
        print(f"[gate] NO baseline at {BASELINE} -- run --update-baseline once on a trusted config first.")
        return 2
    base = json.loads(BASELINE.read_text())
    ok, msg = _verdict(cur, base)
    if a.json:
        print(json.dumps({"pass": ok, "verdict": msg, "current": cur, "baseline": {
            k: base.get(k) for k in ("solved_count", "median_actions_on_solved", "total_actions_on_solved")}}, indent=2))
    else:
        print(f"[gate] current : solved {cur['solved_count']}, median actions/solve "
              f"{cur['median_actions_on_solved']}, timed_out {cur['timed_out_count']}")
        print(f"[gate] baseline: solved {base.get('solved_count')}, median actions/solve "
              f"{base.get('median_actions_on_solved')}")
        print(f"[gate] {'PASS' if ok else 'FAIL'}: {msg}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

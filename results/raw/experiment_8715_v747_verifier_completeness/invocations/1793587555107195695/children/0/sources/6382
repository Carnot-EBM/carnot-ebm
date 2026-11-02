#!/usr/bin/env python3
"""Measure how deep each ARC GameAdapter can actually solve, and pin it as a baseline.

WHY THIS EXISTS (2026-07-31). `ops/arc_solve_registry.yaml` describes each game's solver in
PROSE -- lf52's says "L1-L2: GameAdapter _lf52 ... L3: <a different mechanism>". Nothing
checks that prose against the code. An adapter can silently regress -- a refactor, a changed
action label, a renamed game attribute -- and the registry keeps asserting the old
capability. That is the same shape as the wa30 gap (a claim with no runnable backing) and
the same shape as the guards this project keeps finding narrower than their own docstrings.

WHAT IT DELIBERATELY DOES NOT DO: parse the prose. Extracting "L1-L2" from free text with a
regex is precisely the pattern-matching-narrower-than-the-concept failure being guarded
against. Instead this MEASURES the real depth and records it. The number becomes the claim.

WHY A SCRIPT AND NOT A TEST. Calibration on 2026-07-31 showed a single `solve_adaptered`
call can exceed 10 MINUTES -- it is a verifier-routed best-first search, not a replay. A
24-game sweep therefore belongs in an occasional/milestone-close run, not in pytest. The
fast half is `tests/python/test_arc_adapter_depth.py`, which guards the RECORD this script
writes without re-solving anything.

EACH GAME RUNS IN ITS OWN SUBPROCESS with a hard timeout. A hanging solve must degrade to a
recorded TIMEOUT for that one game, never wedge the sweep -- three prior measurements in
this project died on exactly that.

Usage:
    python scripts/arc_adapter_depth_probe.py                    # all adapters, depth<=3
    python scripts/arc_adapter_depth_probe.py --games tr87,r11l  # a subset
    python scripts/arc_adapter_depth_probe.py --max-level 2 --timeout 600
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

BASELINE = REPO / "ops" / "arc_adapter_depth_baseline.json"

# Run one (game, level) attempt in a child process. Printed as JSON on the last stdout line
# so a noisy ARC SDK logger cannot corrupt the result.
_CHILD = r"""
import json, os, sys, time
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CARNOT_ARC_E3_DIR", "/tmp/adapter_depth_probe/e3")
os.makedirs(os.environ["CARNOT_ARC_E3_DIR"], exist_ok=True)
sys.path.insert(0, {repo!r} + "/python")
sys.path.insert(0, {repo!r} + "/scripts")
import arc_loop_solve as loop
game, target = {game!r}, {level!r}
t0 = time.time()
try:
    res = loop.solve_adaptered(game, target)
    out = {{"ok": True, "reached": int(res.get("reached_level") or 0), "s": round(time.time()-t0, 1)}}
except Exception as exc:
    out = {{"ok": False, "error": f"{{type(exc).__name__}}: {{str(exc)[:160]}}", "s": round(time.time()-t0, 1)}}
print("PROBE_RESULT " + json.dumps(out))
"""


def attempt(game: str, level: int, timeout: int) -> dict:
    """One bounded attempt. Returns {status, reached|error, s}."""
    code = _CHILD.format(repo=str(REPO), game=game, level=level)
    t0 = time.time()
    try:
        proc = subprocess.run(
            [str(REPO / ".venv" / "bin" / "python"), "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "s": timeout}
    for line in reversed((proc.stdout or "").splitlines()):
        if line.startswith("PROBE_RESULT "):
            payload = json.loads(line[len("PROBE_RESULT ") :])
            if payload.get("ok"):
                return {"status": "ok", "reached": payload["reached"], "s": payload["s"]}
            return {"status": "error", "error": payload.get("error"), "s": payload.get("s")}
    return {
        "status": "no_result",
        "s": round(time.time() - t0, 1),
        "stderr_tail": (proc.stderr or "")[-200:],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--games", default="", help="comma-separated; default = every registered adapter"
    )
    ap.add_argument("--max-level", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=900, help="seconds PER (game, level) attempt")
    ap.add_argument("--out", default=str(BASELINE))
    args = ap.parse_args()

    from carnot.agentic import arc_game_adapters as adapters

    games = (
        [g.strip() for g in args.games.split(",") if g.strip()]
        if args.games
        else sorted(adapters._BUILDERS)
    )

    results: dict[str, dict] = {}
    for game in games:
        # Deepest level actually reached. Probe upward and stop at the first level not
        # reached: depth is monotone by construction (reaching L3 requires passing L2), so
        # continuing past a failure would only burn the timeout again.
        depth, detail = 0, []
        for level in range(1, args.max_level + 1):
            r = attempt(game, level, args.timeout)
            detail.append({"level": level, **r})
            print(f"  {game} L{level}: {r}", flush=True)
            if r["status"] == "ok" and r.get("reached", 0) >= level:
                depth = level
                continue
            break
        results[game] = {"measured_depth": depth, "attempts": detail}

    payload = {
        "schema": "carnot.arc_adapter_depth_baseline.v1",
        "what_this_is": (
            "The deepest level each GameAdapter actually reached via solve_adaptered, MEASURED. "
            "It supersedes the registry's prose `solver` field as the checkable claim. A drop "
            "below a recorded depth is an adapter regression."
        ),
        "caveats": (
            "measured_depth is a LOWER BOUND, not the adapter's ceiling: it is capped by "
            "--max-level and by --timeout, and a 'timeout' status means unproven, NOT broken. "
            "Do not read 0 as 'the adapter does not work' without reading its attempts."
        ),
        "max_level_probed": args.max_level,
        "timeout_s_per_attempt": args.timeout,
        "games": results,
    }
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    solid = sum(1 for v in results.values() if v["measured_depth"] >= 1)
    print(f"\n  {solid}/{len(results)} adapters reached at least L1 -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

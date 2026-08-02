#!/usr/bin/env python3
"""Read the row cache mid-flight and print per-arm shape counts. HEALTH MONITORING ONLY.

THIS IS NOT AN INTERIM ANALYSIS AND MUST NOT BECOME A STOPPING RULE. The design (arms,
replicates, the primary, the test) is fixed in out/preregistration.json, written before the
first LLM call. The ONLY stopping rules are the pre-declared job list and the pre-declared wall
budget. Peeking at an effect and then stopping when it looks good is optional stopping, and it
would invalidate the permutation p-value this run reports. What this script is for is catching
a dead server, an arm whose cells are all missing, or a classifier returning None everywhere --
failures that should end the run, not results that should.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent / "out"


def main() -> int:
    rows = [json.loads(p.read_text()) for p in sorted((OUT / "rowcache").glob("*.json"))]
    print(f"{len(rows)} cells cached")
    by: dict[tuple, Counter] = defaultdict(Counter)
    elapsed: dict[tuple, list] = defaultdict(list)
    for r in rows:
        key = (r["stage"], r["arm"])
        by[key][r.get("pred_shape") or "MISSING"] += 1
        elapsed[key].append(r["elapsed_s"])
    for key in sorted(by, key=str):
        c = by[key]
        n = sum(c.values())
        el = elapsed[key]
        print(
            f"  stage{key[0]} {key[1]:4s} n={n:3d} "
            + " ".join(f"{k}={c[k]}" for k in sorted(c))
            + f"  median_s={sorted(el)[len(el) // 2]:.1f}"
        )
    fired = Counter((r["arm"], bool(r.get("goal_only_call_ran"))) for r in rows if r["stage"] == 2)
    if fired:
        print("stage2 goal-only call ran:", dict(fired))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

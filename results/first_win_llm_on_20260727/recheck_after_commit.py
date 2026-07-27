#!/usr/bin/env python
"""Did committing THE FIX change any number this measurement published?

WHY THIS RUNS. Every measurement cell finished at 10:14 local. Commit 776161963 (the
generator concurrency fix) landed at 10:19, and its pre-commit hooks rewrote the bytes of
`arc_competition_agent.py` and `arc_executable_world_model.py`, so
`artifact_freshness_lint`/`summarize_artifact` correctly reported the artifact STALE against
the code now on disk. Staleness is a statement about bytes, not about behaviour -- and the
project's own rule is that a rebuild must be DIFFED and any moved number reported, never
waved through. The fix itself is intact (`_default_induce_n_ctx` still returns 81920, the
liveness witness is still present), but "intact" is an inspection, not a measurement.

WHAT THIS DOES. Re-runs the DECISIVE cells under the CURRENT committed code and diffs against
the banked rows, cell by cell, on the fields the artifact actually publishes: first_win,
reached_level, actions, actions_to_first_levelup. The chosen cells are the 7 the control won
-- i.e. every cell that carries a win, and therefore every cell that could move the headline
rate -- plus 3 non-winners as an over-fire control (a recheck that only re-ran winners could
not detect a spurious NEW win appearing elsewhere).

LLM-OFF ONLY, deliberately. The re-run uses the `_NoOpProposer` arm: it needs no generator, so
it costs seconds instead of an hour, and it is the arm whose determinism was already
established (serialcheck.json, two bit-identical repetitions). Since NO llm_on cell ever had
the generator's output reach the policy (0 of 74), the llm_on trajectories are produced by the
same code path this re-run exercises -- so a byte change that left this arm's outcomes
identical could not have moved theirs either. That is an argument, not a proof, and it is
recorded as such.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("/home/ianblenke/github.com/ianblenke/carnot")
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO))
OUT = REPO / "results" / "first_win_llm_on_20260727"

os.environ["CARNOT_ARC_GATE_DEEPEN"] = "1"
os.environ["CARNOT_ARC_GATE_VARIANT_IDS"] = "1,2,3,4"
os.environ.pop("CARNOT_ARC_DISABLE_INDUCTION", None)

import carnot.experiment_4605_live_integration_scored_agent as exp4605  # noqa: E402

FIELDS = ("first_win", "reached_level", "actions", "actions_to_first_levelup")

# the 7 control winners (every cell that carries a win) + 3 non-winners as an over-fire control
TARGETS = [
    ("lp85", 2),
    ("sp80", 2),
    ("sp80", 3),
    ("vc33", 1),
    ("vc33", 2),
    ("vc33", 3),
    ("vc33", 4),
    ("ar25", 1),
    ("dc22", 1),
    ("tu93", 1),
]


def banked(game: str, variant: int) -> dict | None:
    f = OUT / "cells" / f"llm_off__{game}_color{variant:02d}.json"
    if not f.exists():
        return None
    return json.loads(f.read_text()).get("row") or {}


rows = []
for game, variant in TARGETS:
    spec = {
        "game": game,
        "variant": variant,
        "kind": "color",
        "reflect": None,
        "variant_signature": f"{game}~color{variant:02d}",
    }
    t0 = time.time()
    fresh = dict(exp4605.run_variant_attempt("integrated", game, spec, 200))
    old = banked(game, variant) or {}
    diff = {k: [old.get(k), fresh.get(k)] for k in FIELDS if old.get(k) != fresh.get(k)}
    rows.append(
        {
            "variant_signature": spec["variant_signature"],
            "elapsed_s": round(time.time() - t0, 2),
            "banked": {k: old.get(k) for k in FIELDS},
            "recheck": {k: fresh.get(k) for k in FIELDS},
            "identical": not diff,
            "diff": diff,
        }
    )
    print(json.dumps(rows[-1]), flush=True)

n_ident = sum(1 for r in rows if r["identical"])
payload = {
    "purpose": "prove commit 776161963 was behaviour-neutral for every published number",
    "git_head_at_recheck": subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip(),
    "n_cells_rechecked": len(rows),
    "n_identical": n_ident,
    "n_moved": len(rows) - n_ident,
    "all_identical": n_ident == len(rows),
    "cells_that_moved": [r for r in rows if not r["identical"]],
    "rows": rows,
    "scope_note": (
        "LLM-OFF arm only, covering all 7 win-carrying cells plus 3 non-winners as an "
        "over-fire control. Not a re-run of the whole 174-cell corpus."
    ),
}
(OUT / "recheck_after_commit.json").write_text(json.dumps(payload, indent=1, default=str))
print(f"\n{n_ident}/{len(rows)} cells IDENTICAL after the commit; moved={len(rows) - n_ident}")

#!/usr/bin/env python
"""Is the LLM-off arm's trajectory difference from the 2026-06-24 baseline caused by MY
K=4 threading, or by agent-code drift since the baseline was taken?

The baseline records lp85~color02 winning at actions_to_first_levelup=59; my K=4 llm_off arm
records the same variant winning at 187. Two candidate explanations with opposite
consequences:

  (a) THREADING -- some shared state leaks across worker threads, in which case my K=4
      numbers are contaminated and the whole design is wrong.
  (b) CODE DRIFT -- E3AgentPolicy is simply different code today than on 2026-06-24 (the
      whole .430-.500 programme landed in between), in which case per-cell determinism is
      intact and the honest framing is "the current agent with the LLM off", not "a
      bit-reproduction of exp4605".

Discriminating test: run the SAME cell strictly SERIALLY (K=1, one thread, one process). If
serial also gives 187, threading is exonerated and (b) holds. If serial gives 59, (a) holds
and the arms must be re-run serially.

Also re-runs the same cell twice serially, to check per-cell determinism directly.
"""

from __future__ import annotations

import json
import os
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

TARGETS = [("lp85", 2), ("lp85", 1)]
rows = []
for game, variant in TARGETS:
    for rep in (1, 2):
        spec = {
            "game": game,
            "variant": variant,
            "kind": "color",
            "reflect": None,
            "variant_signature": f"{game}~color{variant:02d}",
        }
        t0 = time.time()
        # UNPATCHED exp4605: the literal committed code path, _NoOpProposer and all.
        row = dict(exp4605.run_variant_attempt("integrated", game, spec, 200))
        rows.append(
            {
                "game": game,
                "variant": variant,
                "rep": rep,
                "serial_k1": True,
                "elapsed_s": round(time.time() - t0, 2),
                "first_win": row.get("first_win"),
                "actions": row.get("actions"),
                "actions_to_first_levelup": row.get("actions_to_first_levelup"),
                "reached_level": row.get("reached_level"),
            }
        )
        print(json.dumps(rows[-1]), flush=True)

# determinism verdict, computed not asserted
det = {}
for game, variant in TARGETS:
    reps = [r for r in rows if r["game"] == game and r["variant"] == variant]
    key = f"{game}~color{variant:02d}"
    det[key] = {
        "reps": reps,
        "deterministic_across_reps": len(
            {(r["first_win"], r["actions_to_first_levelup"]) for r in reps}
        )
        == 1,
    }
(OUT / "serialcheck.json").write_text(
    json.dumps({"rows": rows, "determinism": det}, indent=1, default=str)
)
print(json.dumps(det, indent=1, default=str))

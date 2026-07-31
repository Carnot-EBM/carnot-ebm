#!/usr/bin/env python3
"""PHASE 1 -- read the three budget sweeps and answer the one question they were run for.

THE QUESTION, stated so it cannot be soft-scored: at each completion budget, how many of the
3 attempts produce a USABLE engine -- one that `generate()` would accept AND whose `engine()`
actually returns on every path? `generate_would_accept` alone is the wrong bar: ft09's banked
engine passes it and still returns None on every click.

REPORTED PER BUDGET, per prompt:
  n_accept   -- what generate() gates on today
  n_usable   -- accept AND returns on all paths (the Phase-1 acceptance gate)
  n_hit_cap  -- attempts that consumed the whole budget (stop_type == limit at predicted_n
                == budget). If this stays at 3/3 as the budget doubles, the model is not
                running out of room, it is looping -- the "rambles more" reading.
  mean_ramble/ mean_code_lines -- whether the extra tokens became CODE or became padding.
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DIRS = sys.argv[1:] or ["sweep", "sweep_combined", "sweep_refactor"]

allrows: list[dict] = []
for d in DIRS:
    p = os.path.join(HERE, d, "sweep.json")
    if not os.path.exists(p):
        print(f"  [missing] {p} -- reported as a MISSING OBSERVATION, not a zero")
        continue
    doc = json.load(open(p))
    if doc.get("status") not in ("ok", "partial"):
        print(f"  [blocked] {d}: {doc.get('status')}")
        continue
    if doc.get("status") == "partial":
        print(f"  [PARTIAL] {d}: lane had not finished; {len(doc['rows'])} rows so far")
    allrows.extend(doc["rows"])

by: dict = {}
for r in allrows:
    by.setdefault((r["prompt"], r["budget"]), []).append(r)

hdr = (f"{'prompt':>9} {'budget':>7} {'n':>3} {'accept':>7} {'USABLE':>7} {'hit cap':>8} "
       f"{'mean pred_n':>12} {'mean ramble':>12} {'mean code ln':>13} {'mean wall s':>12}")
print()
print(hdr)
print("-" * len(hdr))
for (pname, budget) in sorted(by, key=lambda k: (k[0], k[1])):
    rows = [r for r in by[(pname, budget)] if r.get("status") == "ok"]
    n = len(rows)
    if not n:
        print(f"{pname:>9} {budget:>7} {'0':>3}  ALL CALLS FAILED AT TRANSPORT LEVEL")
        continue
    acc = sum(1 for r in rows if r.get("generate_would_accept"))
    use = sum(1 for r in rows if r.get("usable_engine"))
    cap = sum(1 for r in rows
              if r.get("stop_type") == "limit" and r.get("predicted_n") == budget)
    mp = sum(r.get("predicted_n") or 0 for r in rows) / n
    mr = sum(r.get("ramble_frac") or 0 for r in rows) / n
    mc = sum(r.get("code_lines") or 0 for r in rows) / n
    mw = sum(r.get("wall_s") or 0 for r in rows) / n
    print(f"{pname:>9} {budget:>7} {n:>3} {acc:>7} {use:>7} {cap:>8} "
          f"{mp:>12.0f} {mr:>12.3f} {mc:>13.1f} {mw:>12.1f}")

print()
print("PER-CALL DETAIL")
for r in sorted(allrows, key=lambda r: (r["prompt"], r["budget"], r["attempt"])):
    if r.get("status") != "ok":
        print(f"  {r['prompt']:>9} b={r['budget']:<6} a={r['attempt']} :: {r.get('status')}")
        continue
    print(f"  {r['prompt']:>9} b={r['budget']:<6} a={r['attempt']} "
          f"stop={str(r.get('stop_type')):>6} pred_n={str(r.get('predicted_n')):>6} "
          f"code_ln={str(r.get('code_lines')):>4} ramble={r.get('ramble_frac'):.3f} "
          f"longest_run={str(r.get('longest_bare_comment_run')):>5} "
          f"accept={str(r.get('generate_would_accept')):>5} "
          f"returns={str(r.get('engine_returns_on_all_paths')):>5} "
          f"USABLE={str(r.get('usable_engine')):>5}")

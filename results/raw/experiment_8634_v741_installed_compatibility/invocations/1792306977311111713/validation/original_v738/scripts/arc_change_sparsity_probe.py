#!/usr/bin/env python3
"""Measure inter-frame change sparsity across the ARC roster.

WHY (2026-07-31). The induce prompt renders FULL 64x64 grids via `to_ascii`, and one of this
session's findings was that only 8 of 25 collected transitions ever fit the 16384-token
budget. If consecutive ARC frames are mostly unchanged, that budget is being spent on
redundancy, and a changed-cell encoding would fit the whole window instead of a third of it.

A single-game spot check on wa30 measured 0.81% of cells changing per transition, and a
changed-cell list at 0.10x the characters of a full-grid render. That is one game. This
sweeps the roster, because a game with global repaints (sk48's manipulators, tn36's program
editor re-layout) could be far denser and would bound how general the saving is.

Prompted by Mage-VL (arXiv 2607.24904), whose codec-native encoder selects motion-salient
patches and reports 1/8-or-less visual token consumption on video. The claim transfers only
if ARC frames are actually sparse in the same way -- hence measuring rather than assuming.

Each game runs in its own SUBPROCESS with a hard timeout: `build_progress_window` runs a real
offline solve, which for some games exceeds 10 minutes, and one hang must not wedge the sweep.

Usage:
    python scripts/arc_change_sparsity_probe.py                  # whole roster
    python scripts/arc_change_sparsity_probe.py --games wa30,sk48 --timeout 300
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

_CHILD = r"""
import json, os, sys
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CARNOT_ARC_E3_DIR", "/tmp/arc_sparsity_probe/e3")
os.makedirs(os.environ["CARNOT_ARC_E3_DIR"], exist_ok=True)
sys.path.insert(0, {repo!r} + "/python")
sys.path.insert(0, {repo!r} + "/scripts")
import numpy as np
from carnot.agentic import arc_actions_to_progress as atp
from carnot.agentic.arc_executable_world_model import to_ascii

game = {game!r}
out = {{"game": game}}
try:
    w = atp.build_progress_window(game)
    if w is None:
        out["status"] = "no_window"
    else:
        window, full, cell = w
        cells = changed = 0
        per = []
        ascii_chars = delta_chars = 0
        for t in full:
            a = np.asarray(t.grid); b = np.asarray(t.next_grid)
            if a.shape != b.shape:
                continue
            n = int(a.size); ch = int((a != b).sum())
            cells += n; changed += ch
            per.append(round(100.0 * ch / n, 3) if n else None)
            # cost of the two encodings, in characters
            ascii_chars += len(to_ascii(a)) + len(to_ascii(b))
            idx = np.where(a != b)
            delta_chars += len(str([(int(r), int(c), int(b[r, c])) for r, c in zip(*idx)]))
        out.update({{
            "status": "ok",
            "n_transitions": len(full),
            "window_len": len(window),
            "grid_cells": int(np.asarray(full[0].grid).size) if full else None,
            "pct_changed_mean": round(100.0 * changed / cells, 3) if cells else None,
            "pct_changed_max": max([p for p in per if p is not None], default=None),
            "full_ascii_chars": ascii_chars,
            "delta_chars": delta_chars,
            "delta_ratio": round(delta_chars / ascii_chars, 4) if ascii_chars else None,
        }})
except Exception as exc:
    out["status"] = "error"
    out["error"] = f"{{type(exc).__name__}}: {{str(exc)[:160]}}"
print("SPARSITY_RESULT " + json.dumps(out))
"""


def probe(game: str, timeout: int) -> dict:
    code = _CHILD.format(repo=str(REPO), game=game)
    try:
        proc = subprocess.run(
            [str(REPO / ".venv" / "bin" / "python"), "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {"game": game, "status": "timeout"}
    for line in reversed((proc.stdout or "").splitlines()):
        if line.startswith("SPARSITY_RESULT "):
            return json.loads(line[len("SPARSITY_RESULT ") :])
    return {"game": game, "status": "no_result", "stderr_tail": (proc.stderr or "")[-160:]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="")
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--out", default=str(REPO / "results" / "arc_change_sparsity_20260731.json"))
    args = ap.parse_args()

    from carnot.agentic import arc_game_adapters as adapters

    games = (
        [g.strip() for g in args.games.split(",") if g.strip()]
        if args.games
        else sorted(adapters._BUILDERS)
    )

    rows = []
    print(f"{'game':<7}{'status':<11}{'%chg mean':>10}{'%chg max':>10}{'delta/ascii':>13}")
    for g in games:
        r = probe(g, args.timeout)
        rows.append(r)
        if r.get("status") == "ok":
            print(
                f"{g:<7}{'ok':<11}{r['pct_changed_mean']:>9.2f}%{r['pct_changed_max']:>9.2f}%"
                f"{r['delta_ratio']:>13.3f}"
            )
        else:
            print(f"{g:<7}{r.get('status'):<11}{'':>10}{'':>10}{'':>13}")

    ok = [r for r in rows if r.get("status") == "ok"]
    agg = {
        "schema": "carnot.arc_change_sparsity.v1",
        "question": (
            "Are consecutive ARC-AGI-3 frames sparse enough that a changed-cell encoding would "
            "materially shrink the induce prompt? The prompt renders full 64x64 grids and only "
            "8 of 25 transitions fit the 16384-token budget."
        ),
        "prompted_by": "Mage-VL (arXiv 2607.24904) codec-native motion-salient patch selection",
        "caveat": (
            "delta_ratio compares CHARACTER counts of two encodings, not tokens, and says "
            "nothing about whether an inducer can learn as well from deltas as from full "
            "grids. It bounds the possible saving; it does not demonstrate one."
        ),
        "n_games_measured": len(ok),
        "n_games_attempted": len(rows),
        "pct_changed_mean_over_games": (
            round(sum(r["pct_changed_mean"] for r in ok) / len(ok), 3) if ok else None
        ),
        "delta_ratio_mean_over_games": (
            round(sum(r["delta_ratio"] for r in ok) / len(ok), 4) if ok else None
        ),
        "delta_ratio_worst": max((r["delta_ratio"] for r in ok), default=None),
        "games": rows,
    }
    Path(args.out).write_text(json.dumps(agg, indent=2) + "\n")
    if ok:
        print(
            f"\n  mean changed: {agg['pct_changed_mean_over_games']}%  |  "
            f"mean delta/ascii: {agg['delta_ratio_mean_over_games']}  |  "
            f"worst: {agg['delta_ratio_worst']}"
        )
    print(f"  {len(ok)}/{len(rows)} measured -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

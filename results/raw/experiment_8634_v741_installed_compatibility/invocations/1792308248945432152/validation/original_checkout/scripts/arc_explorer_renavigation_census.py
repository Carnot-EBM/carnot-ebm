#!/usr/bin/env python3
"""Roster-wide decomposition of an ARC exploration run's NON-EXPANDING actions.

Driver for `scripts/arc_explorer_renavigation_probe.py`: one killable subprocess per
(game, arm), then aggregate into named action classes with shares, per-class avoidability
with a NAMED mechanism, and the theoretical prize converted into ARC-AGI-3's own scoring
units (`min((baseline/agent)**2, 115)`).

Two arms, both LLM-off and induction-disabled so the comparison is deterministic and
generator-independent:

  * ``shipped``      -- the configuration the submitted agent ships (this is the census).
  * ``tier_off``     -- identical except ``CARNOT_ARC_FRONTIER_TIER_EXHAUSTION=0``. This is a
                        MECHANISM PROBE, not a proposed change: it tests whether the largest
                        overhead class is caused by the global tier barrier or by the
                        environment's reset-only semantics. The barrier is shipped ON because
                        it won +2..+4 games on click games (REQ-ARC-WMTE-5836), so a cheaper
                        arm here is evidence about CAUSE, never on its own a reason to flip it.

Never plays a scored or online game; never starts a generator; never touches a GPU.

Usage:
    CARNOT_ARC_E3_DIR=<scratch> python scripts/arc_explorer_renavigation_census.py \
        --out-dir results/arc_explorer_renavigation_20260802
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent

# The named classes, each with the mechanism that would have to change for those actions not
# to be spent. "Avoidable" here means A MECHANISM CAN BE NAMED -- not that flipping it is
# free, and not that it has been shown to help.
CLASS_MECHANISM: dict[str, dict[str, Any]] = {
    "bootstrap.reset": {
        "avoidable": False,
        "mechanism": (
            "None. The first RESET is how the agent obtains its first frame; there is no "
            "cheaper way to start an episode."
        ),
    },
    "expansion.probe_discovered_new_state": {
        "avoidable": False,
        "mechanism": (
            "None -- this IS the work. One action spent, one previously-unknown state added "
            "to the agent's graph."
        ),
    },
    "expansion.probe_revisited_known_state": {
        "avoidable": False,
        "mechanism": (
            "Not avoidable by navigation policy: the action was UNTESTED at that state and it "
            "DID move the board, so taking it is genuine information (it adds a transition "
            "edge) even though the resulting state was already in the graph. Reducible only "
            "by a transition model good enough to predict the destination before paying for "
            "it -- a generation-side lever, and the one every other probe this session found "
            "inert."
        ),
    },
    "expansion.probe_was_inert_frame_unchanged": {
        "avoidable": True,
        "mechanism": (
            "An inert-action PRIOR that predicts 'this action will not change the frame' "
            "before the action is spent. The machinery exists and is shipped ON: "
            "`inert_click_pruner` (SUBMITTED_INERT_CLICK_PRUNER_ENABLED), "
            "`frame_change_scorer` / `ActionEffectExpansionPrior`, "
            "`prune_arc_actions_by_prior_quantile`. These actions are what it did NOT catch. "
            "Avoidable in the sense that a named mechanism targets them; NOT demonstrated "
            "avoidable, since no arm here shows a better prior actually recovering them."
        ),
    },
    "renavigation.reset_action": {
        "avoidable": True,
        "mechanism": (
            "Do not leave a node while it still has untested work the agent is willing to "
            "spend on. Every RESET in these runs was the first step of a plan to return to a "
            "node the agent had ALREADY STOOD ON. It is a consequence of the frontier's "
            "departure decision, not of the state space: the environment offers no undo, so "
            "once the agent decides to be somewhere else, RESET+replay is the only route."
        ),
    },
    "renavigation.replay_prefix_shared_with_current_path": {
        "avoidable": True,
        "mechanism": (
            "Same as the RESET it follows. These steps re-walk a prefix of the path the agent "
            "was standing on -- ground it had covered minutes earlier in the same episode. "
            "They disappear if the departure does not happen, and only if: with reset-only "
            "semantics there is no cheaper way to reach an ANCESTOR of your current state."
        ),
    },
    "renavigation.replay_suffix_past_divergence": {
        "avoidable": True,
        "mechanism": (
            "Order the frontier for locality, or record enough forward edges that "
            "`_exact_shortest_path` / `_partial_forward_path` can walk there without a RESET. "
            "These steps cross ground the current path does NOT cover, so they are the part of "
            "a replay that buys real distance."
        ),
    },
    "renavigation.forward_walk_no_reset": {
        "avoidable": True,
        "mechanism": (
            "Frontier ordering for locality. These are already the CHEAP form of navigation "
            "(no RESET); shortening them means choosing a nearer frontier target."
        ),
    },
    "plan.execute_step": {
        "avoidable": False,
        "mechanism": (
            "Not a search cost. Zero in these runs by construction (induction disabled)."
        ),
    },
    "other": {"avoidable": False, "mechanism": "Unclassified residue; reported, not explained."},
}

RENAVIGATION_CLASSES = (
    "renavigation.reset_action",
    "renavigation.replay_prefix_shared_with_current_path",
    "renavigation.replay_suffix_past_divergence",
    "renavigation.forward_walk_no_reset",
)


def score(baseline_actions: float, agent_actions: float) -> float:
    """ARC-AGI-3's per-level score: min((baseline/agent)**2, 115)."""

    if agent_actions <= 0:
        return 115.0
    return min((float(baseline_actions) / float(agent_actions)) ** 2, 115.0)


def run_arm(game: str, arm: str, out_dir: Path, budget: int, seed: int, scratch: Path) -> Path:
    out = out_dir / "cells" / f"{game}__{arm}.json"
    env = dict(os.environ)
    env["CARNOT_ARC_E3_DIR"] = str(scratch / "e3" / arm)
    env["CARNOT_ARC_DISABLE_INDUCTION"] = "1"
    # No generator is constructed by the probe, but an empty visible-device list makes the
    # "this run touched no GPU" claim structural rather than a promise.
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["PYTHONPATH"] = str(REPO / "python")
    if arm == "tier_off":
        env["CARNOT_ARC_FRONTIER_TIER_EXHAUSTION"] = "0"
    (scratch / "e3" / arm).mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(REPO / "scripts" / "arc_explorer_renavigation_probe.py"),
        "--game",
        game,
        "--seed",
        str(seed),
        "--budget",
        str(budget),
        "--out",
        str(out),
    ]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {
                    "game": game,
                    "arm": arm,
                    "error": f"exit {proc.returncode}",
                    "stderr_tail": proc.stderr[-2000:],
                },
                indent=1,
            )
        )
    return out


def aggregate(cells: list[dict[str, Any]]) -> dict[str, Any]:
    total = 0
    counts: dict[str, int] = {}
    for cell in cells:
        for k, v in (cell.get("class_counts") or {}).items():
            counts[k] = counts.get(k, 0) + int(v)
            total += int(v)
    classes = []
    for k in sorted(counts, key=lambda x: -counts[x]):
        meta = CLASS_MECHANISM.get(k, CLASS_MECHANISM["other"])
        classes.append(
            {
                "kind": k,
                "actions": counts[k],
                "share_of_budget": round(counts[k] / total, 6) if total else 0.0,
                "avoidable": bool(meta["avoidable"]),
                "why": meta["mechanism"],
            }
        )
    return {"total_actions": total, "class_counts": counts, "classes": classes}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--budget", type=int, default=240)
    ap.add_argument("--seed", type=int, default=20260802)
    ap.add_argument("--games", default="")
    ap.add_argument("--arms", default="shipped,tier_off")
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

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    t0 = time.time()
    cells: dict[str, list[dict[str, Any]]] = {a: [] for a in arms}
    for arm in arms:
        for game in games:
            path = run_arm(game, arm, out_dir, args.budget, args.seed, scratch)
            try:
                cell = json.loads(path.read_text())
            except Exception as exc:
                cell = {"game": game, "error": f"unreadable:{exc}"}
            cell["arm"] = arm
            cells[arm].append(cell)
            print(
                f"[census] {arm}/{game}: "
                f"{cell.get('actions_recorded')} actions, "
                f"{cell.get('n_navigation_episodes')} nav episodes",
                flush=True,
            )

    summary = {arm: aggregate([c for c in cells[arm] if not c.get("error")]) for arm in arms}
    payload = {
        "games": games,
        "arms": arms,
        "budget": args.budget,
        "seed": args.seed,
        "duration_s": round(time.time() - t0, 3),
        "summary": summary,
        "cells": {arm: cells[arm] for arm in arms},
    }
    raw = json.dumps(payload, sort_keys=True, default=str).encode()
    payload["reproducibility_checksum"] = "sha256:" + hashlib.sha256(raw).hexdigest()
    (out_dir / "census.json").write_text(json.dumps(payload, indent=1, default=str))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

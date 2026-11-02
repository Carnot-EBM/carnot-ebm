"""Config Layer B BREADTH SWEEP -- run the scaffolded LLM rule-inducer across the config-class games to
measure how many GROUND. Move-2 (arc3_config_layerb_scaffolded.py) grounded ka59 (Tier 2) via structured
extraction; tr87 (rewrite class) is a known-mechanic outlier handled by a from-pixels glyph decoder. This
sweep tests whether the scaffold GENERALIZES across the other count/relation-class config games -- the
~13-game config wall behind FAIL_EXPLORATION.

Reuses the scaffolded harness wholesale (collect / build_prompt / _generate_bounded / verify /
_looks_literal_hardcode) but loads the local gemma-12B ONCE and reuses the server across games (the
per-game harness would reload the 4 GB model each time). Only games with a BANKED win can be grounded
(the verifier needs ground truth); no-banked config games are recorded blocked. iGPU port 8920, offline,
zero quota.

Usage: python scripts/experiments/arc3_config_layerb_sweep.py [game1 game2 ...]
Default sweep = the config-class games with/without banked wins, excluding ka59 (done) + tr87 (rewrite)."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

from carnot.agentic import arc_executable_world_model as e3

# config-class candidates (banked: sc25/tn36/cd82/wa30 -> groundable; no-bank: bp35/dc22/g50t/lf52/s5i5)
DEFAULT_GAMES = ["sc25", "tn36", "cd82", "wa30", "bp35", "dc22", "g50t", "lf52", "s5i5"]


def _scaffold():
    spec = importlib.util.spec_from_file_location(
        "sf", str(REPO / "scripts" / "experiments" / "arc3_config_layerb_scaffolded.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def run_one(sf, proposer, game):
    sf.GAME = game                                       # collect()/build_prompt() read the module global
    try:
        scene, eb, bg, ref_box, win, ch, nonwins = sf.collect()
    except Exception as ex:
        return {"game": game, "tier": "ERROR", "honest_verdict": f"collect_error_{type(ex).__name__}", "detail": str(ex)[:120]}
    if eb is None or win is None or len(nonwins) < 2:
        return {"game": game, "tier": "BLOCKED",
                "honest_verdict": "blocked_no_editable_or_no_win_or_too_few_nonwins",
                "has_editable": eb is not None, "has_win": win is not None, "n_nonwins": len(nonwins)}
    win_sub = sf._edit_sub(win, eb)
    row = {"game": game, "editable_cells": int(win_sub.size), "n_nonwins": len(nonwins)}
    ok, code, raw = sf._generate_bounded(proposer, sf.build_prompt(scene, eb, bg, ref_box, win, nonwins))
    row["coherent_runnable"] = bool(ok)
    if not ok:
        row["tier"] = "TIER0_FAIL"; row["honest_verdict"] = "tier0_fail_no_coherent_predicate"; row["msg"] = str(code)[:120]
        return row
    ns = {}
    try:
        exec(code, ns)  # noqa: S102 -- inducing a verifier predicate; sandboxed-by-scope
        v = sf.verify(ns.get("is_win", lambda g: False), win, nonwins)
        fpr = v.get("false_positive_rate")
        grounded = bool(v.get("fires_on_win")) and fpr is not None and fpr < 0.2
        literal = sf._looks_literal_hardcode(code, win_sub)
        row["verification"] = v; row["rule_grounded"] = grounded; row["literal_hardcode"] = bool(literal)
        row["tier"] = ("TIER2_GROUNDED_RELATIONAL" if grounded and not literal else
                       "TIER1_GROUNDED_LITERAL" if grounded else "TIER0_COHERENT_NOT_GROUNDED")
        row["honest_verdict"] = f"complete_scaffolded_{row['tier'].lower()}"
    except Exception as ex:
        row["tier"] = "TIER0_UNCOMPILABLE"; row["exec_error"] = f"{type(ex).__name__}: {str(ex)[:100]}"
        row["honest_verdict"] = "tier0_coherent_but_uncompilable"
    return row


def main():
    games = sys.argv[1:] or DEFAULT_GAMES
    print(f"== CONFIG LAYER B BREADTH SWEEP over {games} ==", flush=True)
    sf = _scaffold()
    proposer = e3.LocalGGUFProposer(repo_substr="gemma-4-12B-it", port=8920, timeout=600)
    rows = []
    try:
        for g in games:
            r = run_one(sf, proposer, g)
            rows.append(r)
            print(f"  {g:6} -> {r.get('tier'):28} {r.get('honest_verdict')}", flush=True)
    finally:
        try:
            proposer.stop()
        except Exception:
            pass
    grounded = [r["game"] for r in rows if r.get("rule_grounded")]
    coherent = [r["game"] for r in rows if r.get("coherent_runnable")]
    blocked = [r["game"] for r in rows if r.get("tier") == "BLOCKED"]
    out = {"experiment": "arc3_config_layerb_sweep", "games": games,
           "n_grounded": len(grounded), "grounded_games": grounded,
           "n_coherent_reading_fixed": len(coherent), "coherent_games": coherent,
           "blocked_no_banked_win": blocked, "per_game": rows,
           "ka59_reference": "tier2_grounded (prior run)", "tr87_reference": "rewrite-class, from-pixels decoder",
           "honest_verdict": (f"complete_config_sweep_grounded_{len(grounded)}_of_{len([r for r in rows if r.get('tier') not in ('BLOCKED','ERROR')])}"
                              f"_groundable_coherent_{len(coherent)}"),
           "inference_substrate": "offline_arc_agi3_layerb_scaffolded_sweep_iGPU_port8920"}
    (REPO / "results" / "arc3_config_layerb_sweep.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\n  GROUNDED: {grounded}", flush=True)
    print(f"  coherent (reading-fixed): {coherent}", flush=True)
    print(f"  blocked (no banked win): {blocked}", flush=True)
    print(f"  -> {out['honest_verdict']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

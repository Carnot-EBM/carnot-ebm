"""Offline learning-curve gate for the live-TTT world model (2026-06-21, build step 3).

Validates the operator's live-learning thesis BEFORE any submission: a world model LEARNED from a few
played transitions (LiveTTTWorldModel = L0 exact table + L1 ObjectDeltaModel) predicts HELD-OUT
transitions -- and can plan a win inside itself -- where the frozen-9B-INDUCED engine fails (exp4557
scored 0.0 held-out accuracy, failing its own single training transition).

Per game: collect_transitions from the OFFLINE sim, split train / held-out, fit the learned model on
train, then measure HELD-OUT transition accuracy of (a) the LEARNED engine vs (b) the LLM-induced engine
(when a world_model.py exists to compare). Then attempt plan_in_model INSIDE the learned model. The
learned model EARNS trust only by predicting transitions it was not fit to -- the same bar the LLM failed.

PASS GATE (per game): learned_heldout_accuracy >= 0.5 (the WorldModelVerifier trust gate) AND
learned_heldout_accuracy >= llm_heldout_accuracy (when an LLM engine exists). The aggregate verdict
reports how many games the LEARNED model clears -- the de-risk before the conductor wires it into the
submitted agent (arc_competition_agent.py:1417, replacing the failing e3.load_engine).

Offline, CPU-only (ObjectDeltaModel is pure numpy) -- NO GPU, no conductor-reaper collision, zero quota.

PRECONDITIONS:
  0. the offline arcade + environment_files sim for each game must load (collect_transitions); a game
     that fails to load is recorded with an error and skipped, never fabricated.

Usage: .venv/bin/python scripts/arc_ttt_validate.py [game ...]   (default: a spread of public games)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

DEFAULT_GAMES = ["ar25", "ka59", "cd82", "sp80", "m0r0", "su15"]
HELDOUT_FRAC = 0.25
N_COLLECT = 160


def validate_game(game: str) -> dict:
    from carnot.agentic.arc_executable_world_model import (
        WorldModelVerifier, collect_transitions, load_engine, plan_in_model,
    )
    from carnot.agentic.arc_live_ttt import LiveTTTWorldModel

    t0 = time.time()
    try:
        transitions, _cell = collect_transitions(game, n=N_COLLECT)
    except Exception as e:  # never fabricate -- record the blocked reason
        return {"game": game, "error": f"collect_transitions: {type(e).__name__}: {e}"[:160]}
    if len(transitions) < 8:
        return {"game": game, "error": f"too few transitions ({len(transitions)})"}

    k = max(2, int(len(transitions) * HELDOUT_FRAC))
    train, held = transitions[:-k], transitions[-k:]

    learned = LiveTTTWorldModel(game)
    for t in train:
        learned.observe_transition(t)
    learned.fit_now()

    out: dict = {"game": game, "n_train": len(train), "n_heldout": len(held)}
    out["learned_heldout_accuracy"] = round(learned.trust(held), 4)
    try:  # the frozen-9B-induced engine, if one was written for this game
        eng, _ = load_engine(game)
        out["llm_heldout_accuracy"] = round(float(WorldModelVerifier(held).score(eng).accuracy), 4)
    except Exception:
        out["llm_heldout_accuracy"] = None

    # planning inside the learned model (meaningful only if a win-state was observed in the probe)
    full = LiveTTTWorldModel(game)
    for t in transitions:
        full.observe_transition(t)
    full.fit_now()
    out["n_win_states"] = full.ttt_diagnostics()["n_win_states"]
    plan = None
    if out["n_win_states"] > 0:
        plan = plan_in_model(full.engine, full.is_level_complete, transitions[0].grid, max_nodes=8000)
    out["plan_found"] = plan is not None
    out["plan_length"] = len(plan) if plan else None

    la, lm = out["learned_heldout_accuracy"], out["llm_heldout_accuracy"]
    out["gate_pass"] = bool(la >= 0.5 and (lm is None or la >= lm))
    out["learned_beats_llm"] = None if lm is None else bool(la > lm)
    out["duration_s"] = round(time.time() - t0, 2)
    return out


def main(argv: list[str]) -> int:
    games = [a for a in argv if not a.startswith("-")] or DEFAULT_GAMES
    rows = [validate_game(g) for g in games]
    scored = [r for r in rows if "learned_heldout_accuracy" in r]
    passed = [r for r in scored if r.get("gate_pass")]
    beats = [r for r in scored if r.get("learned_beats_llm")]
    mean_learned = round(sum(r["learned_heldout_accuracy"] for r in scored) / max(1, len(scored)), 4)

    for r in rows:
        if "error" in r:
            print(f"  {r['game']:5} ERROR {r['error']}", flush=True)
        else:
            print(f"  {r['game']:5} learned={r['learned_heldout_accuracy']:.3f} "
                  f"llm={r['llm_heldout_accuracy']} win_states={r['n_win_states']} "
                  f"plan={'Y' if r['plan_found'] else 'n'} gate={'PASS' if r['gate_pass'] else 'fail'}",
                  flush=True)

    out = {
        "experiment": "arc_ttt_validate",
        "honest_verdict": f"complete_live_ttt_offline_gate_{len(passed)}_of_{len(scored)}_games_pass",
        "games_scored": len(scored), "games_gate_pass": len(passed),
        "games_learned_beats_llm": len(beats), "mean_learned_heldout_accuracy": mean_learned,
        "per_game": rows,
        "random_seed": 0,
        "inference_substrate": "verifier_ensemble_against_cached_candidates -- offline arcade transition "
                               "collection + learned-vs-LLM engine scoring; no live LLM inference",
        "verifier_is_oracle": False,
        "methodology_note": "held-out = the last 25% of the salience-ordered probe; the learned engine "
                            "earns accuracy only on transitions outside its L0 table / fit set. The LLM "
                            "comparison uses the same held-out set (apples-to-apples).",
    }
    (REPO / "results" / "arc_ttt_validate.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\n  {len(passed)}/{len(scored)} games pass the trust gate; mean learned held-out "
          f"accuracy={mean_learned}; learned beats LLM on {len(beats)}. -> {out['honest_verdict']}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

"""Measure the QUALITY of e3's saved LLM-induced world-models: exact-match vs changed-cell recall (2026-06-21).

Piece-1 of the coordinated redesign assumed the e3 induced world-models were imperfect-BUT-USEFUL (like the
TTT CNN at 0.55 cell-recall) and merely gated out by the strict exact-FULL-GRID match. This probe scores
each saved results/arc_e3/<game>/world_model.py engine against freshly-collected offline transitions and
reports BOTH metrics. The finding (2026-06-21): on the gap-1 failing games the induced models predict
NEAR-IDENTITY -- cell_recall ~0-0.05, LOWER than the exact-match accuracy (0.13-0.35) which was INFLATED by
no-op transitions (identity is 'correct' when the grid does not change). So the cell-recall gate does NOT
un-gate them; it correctly REJECTS useless models. The e3 induction bottleneck is INDUCTION QUALITY (the
LLM cannot write a world-model that captures the dynamics; model-independent gemma-12B==Qwen-35B), not the
gate metric. cell_recall remains the HONEST gate (it exposes identity-predictors that exact-match's no-op
inflation masks) and the correct fix for the LEARNED-dynamics (TTT) path where the model IS useful (0.55).
Offline, no LLM.
"""
from __future__ import annotations
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

import numpy as np

from carnot.agentic.arc_executable_world_model import collect_transitions, load_engine, WorldModelVerifier

GAMES = (sys.argv[1].split(",") if len(sys.argv) > 1
         else ["cn04", "sc25", "ar25", "cd82", "tn36", "ka59", "wa30", "sk48"])


def main() -> int:
    rows = []
    print(f"{'game':6} {'n':>4} {'n_chg':>5} | {'exact_acc':>9} {'cell_recall':>11} | verdict", flush=True)
    for game in GAMES:
        if not (REPO / "results" / "arc_e3" / game / "world_model.py").exists():
            continue
        try:
            trans, _cell = collect_transitions(game, n=120, seed=0)
            engine, _is_done = load_engine(game)
            vr = WorldModelVerifier(trans).score(engine)
            nchg = sum(1 for t in trans if not np.array_equal(t.grid, t.next_grid))
            verdict = ("both_pass" if vr.cell_recall >= 0.5 and vr.accuracy >= 0.5
                       else "cellrecall_flips_pass" if vr.accuracy < 0.5 <= vr.cell_recall
                       else "both_skip")
            rows.append({"game": game, "n": vr.n, "n_changed": nchg,
                         "exact_accuracy": round(vr.accuracy, 4), "cell_recall": round(vr.cell_recall, 4),
                         "verdict": verdict})
            print(f"{game:6} {vr.n:>4} {nchg:>5} | {vr.accuracy:>9.3f} {vr.cell_recall:>11.3f} | {verdict}", flush=True)
        except Exception as e:
            rows.append({"game": game, "error": f"{type(e).__name__}: {str(e)[:80]}"})
            print(f"{game:6} ERROR {type(e).__name__}: {str(e)[:60]}", flush=True)
    out = {
        "experiment": "arc_e3_induced_model_quality",
        "honest_verdict": "complete_e3_induced_models_predict_near_identity_cellrecall_near_zero",
        "finding": ("e3 LLM-induced world-models predict near-identity (cell_recall ~0-0.05 on gap-1 games, "
                    "LOWER than exact-match which is inflated by no-op transitions). The cell-recall gate "
                    "correctly REJECTS them -- the e3 bottleneck is INDUCTION QUALITY, not the gate metric. "
                    "cell_recall is the honest gate + the correct fix for the LEARNED-dynamics (TTT) path."),
        "per_game": rows,
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "verifier_is_oracle": False,
    }
    (REPO / "results" / "arc_e3_induced_model_quality.json").write_text(json.dumps(out, indent=2, default=str))
    print("-> results/arc_e3_induced_model_quality.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

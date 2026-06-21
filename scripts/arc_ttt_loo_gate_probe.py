"""Gate-readiness probe: does the prior-warmstarted TTT trust gate FIRE on games the prior never saw?

The decisive gate signal (2026-06-21). The live agent only USES the per-game TTT world-model when its
held-out WorldModelVerifier accuracy (% of held-out transitions reproduced EXACTLY) clears trust_threshold
(0.5). So the question "does the warm-start prior move our hidden-game score" reduces to: on the 5 LOO games
the prior NEVER trained on, does warm-starting from the prior let the per-game fit clear the 0.5 gate (so the
TTT path activates on an unseen game) -- vs scratch (no prior)?

Gate-only (no planning BFS -- that is secondary and expensive); GPU; flush per game. Offline, zero-quota.
"""
from __future__ import annotations
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

import json

from carnot.agentic.arc_transition_capture import TransitionCorpus
from carnot.agentic.arc_live_ttt import gated_engine_from_transitions

LOO = ["cd82", "ka59", "sc25", "tn36", "lp85"]  # the 5 games the prior NEVER trained on (true transfer surface)
NOPRIOR = "models/__nonexistent_prior__.pt"     # forces scratch (no warm-start)


def probe(game, tr, prior_path):
    eng, _isdone, diag = gated_engine_from_transitions(game, tr, prior_path=prior_path, trust_threshold=0.5)
    return {"gate": diag.get("gate"), "fired": eng is not None, "skip": diag.get("skip"),
            "heldout_acc": (round(diag["heldout_accuracy"], 4) if diag.get("heldout_accuracy") is not None else None)}


def main() -> int:
    corpus = TransitionCorpus()
    rows = []
    warm_fire = scratch_fire = 0
    print("=== TTT trust-gate (0.5 exact-match) on the 5 LOO held-out games: WARM(prior) vs SCRATCH ===", flush=True)
    print(f"{'game':6} {'n_tr':>5} | {'WARM':>6} {'acc':>7} | {'SCRATCH':>7} {'acc':>7}", flush=True)
    for g in LOO:
        tr = corpus.load(g)
        w = probe(g, tr, "models/arc_dynamics_prior.pt")
        s = probe(g, tr, NOPRIOR)
        warm_fire += int(w["fired"]); scratch_fire += int(s["fired"])
        rows.append({"game": g, "n_tr": len(tr), "warm": w, "scratch": s})
        print(f"{g:6} {len(tr):>5} | {str(w['fired']):>6} {str(w['heldout_acc']):>7} | "
              f"{str(s['fired']):>7} {str(s['heldout_acc']):>7}", flush=True)
    print(f"\nGATE-FIRES on unseen games: WARM {warm_fire}/5  vs  SCRATCH {scratch_fire}/5", flush=True)
    out = {
        "experiment": "arc_ttt_loo_gate_probe",
        "honest_verdict": f"complete_ttt_loo_gate_warm_{warm_fire}_scratch_{scratch_fire}_of_5",
        "trust_threshold": 0.5, "metric": "WorldModelVerifier exact-grid-match accuracy on within-game held-out split",
        "warm_gate_fires": warm_fire, "scratch_gate_fires": scratch_fire, "per_game": rows,
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "verifier_is_oracle": False,
        "interpretation": ("Gate must FIRE for the prior+TTT path to activate on a hidden game; if warm fires ~0/5 "
                           "the path stays dormant on unseen games and the prior cannot move the 0.08 hidden score."),
    }
    (REPO / "results" / "arc_ttt_loo_gate_probe.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"-> {out['honest_verdict']} (results/arc_ttt_loo_gate_probe.json)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

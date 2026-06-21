"""Gate-readiness probe: does the prior-warmstarted TTT trust gate FIRE on games the prior never saw?

The decisive gate signal (2026-06-21). The live agent only USES the per-game TTT world-model when its
held-out trust metric clears trust_threshold (0.5). v1 of this probe found the EXACT-full-grid-match gate
fires 0/5 on the LOO held-out games (a 64x64 CNN is ~55% changed-cell-accurate but ~0% exact-grid). v2 adds
the CELL-RECALL trust metric (changed-cell recall, matched to the model's granularity) and compares: does
re-metricing the gate make the prior+TTT path fire on unseen games? Reports raw values for both metrics so
any threshold choice is auditable. Offline, zero-quota.
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
THRESH = 0.5


def probe(game, tr, prior_path, metric):
    eng, _isdone, diag = gated_engine_from_transitions(
        game, tr, prior_path=prior_path, trust_threshold=THRESH, trust_metric=metric)
    return {"gate": diag.get("gate"), "fired": eng is not None,
            "exact": diag.get("heldout_accuracy"), "cell_recall": diag.get("heldout_cell_recall")}


def main() -> int:
    corpus = TransitionCorpus()
    rows = []
    fires = {"exact_warm": 0, "exact_scratch": 0, "cell_warm": 0, "cell_scratch": 0}
    print(f"=== TTT trust-gate on 5 LOO held-out games (threshold {THRESH}); EXACT vs CELL-RECALL ===", flush=True)
    print(f"{'game':6} {'n_tr':>5} | {'exact':>6} | {'cellrec_warm':>12} {'fires':>5} | "
          f"{'cellrec_scr':>11} {'fires':>5}", flush=True)
    for g in LOO:
        tr = corpus.load(g)
        ew = probe(g, tr, "models/arc_dynamics_prior.pt", "exact")
        cw = probe(g, tr, "models/arc_dynamics_prior.pt", "cell_recall")
        cs = probe(g, tr, NOPRIOR, "cell_recall")
        fires["exact_warm"] += int(ew["fired"])
        fires["cell_warm"] += int(cw["fired"])
        fires["cell_scratch"] += int(cs["fired"])
        rows.append({"game": g, "n_tr": len(tr), "exact_warm": ew, "cell_warm": cw, "cell_scratch": cs})
        print(f"{g:6} {len(tr):>5} | {str(ew['exact']):>6} | "
              f"{str(cw['cell_recall']):>12} {str(cw['fired']):>5} | "
              f"{str(cs['cell_recall']):>11} {str(cs['fired']):>5}", flush=True)
    print(f"\nGATE-FIRES (threshold {THRESH}):", flush=True)
    print(f"  EXACT gate, warm:        {fires['exact_warm']}/5   (the current live-agent gate)", flush=True)
    print(f"  CELL-RECALL gate, warm:  {fires['cell_warm']}/5   <- does re-metricing make the path fire?", flush=True)
    print(f"  CELL-RECALL gate, scratch:{fires['cell_scratch']}/5  (no prior -- isolates the prior's lift)", flush=True)
    out = {
        "experiment": "arc_ttt_loo_gate_probe_v2_cellrecall",
        "honest_verdict": (f"complete_ttt_loo_gate_exact_warm_{fires['exact_warm']}_"
                           f"cellrecall_warm_{fires['cell_warm']}_scratch_{fires['cell_scratch']}_of_5"),
        "trust_threshold": THRESH,
        "exact_gate_fires_warm": fires["exact_warm"],
        "cell_recall_gate_fires_warm": fires["cell_warm"],
        "cell_recall_gate_fires_scratch": fires["cell_scratch"],
        "per_game": rows,
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "verifier_is_oracle": False,
        "interpretation": ("If cell-recall-warm fires materially more than exact-warm AND more than "
                           "cell-recall-scratch, the prior+TTT path can be ACTIVATED on unseen games by "
                           "re-metricing the gate. Firing is necessary-not-sufficient: whether a plan BFS'd "
                           "through a ~0.6-cell-recall model actually SOLVES (+ goal-induction) is the next "
                           "gate signal, not settled here."),
    }
    (REPO / "results" / "arc_ttt_loo_gate_probe.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"-> {out['honest_verdict']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

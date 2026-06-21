"""Pretrain the cross-game mechanic PRIOR + measure few-shot transfer (the pretrain-prior path, 2026-06-21).

Trains a single CNNDynamics on the transition corpus (human play + probes, the 25 public 64x64 games) to
learn a GENERAL prior over ARC-AGI-3 mechanics, then tests whether a per-game learner WARM-STARTED from the
prior reaches state-CHANGING transition accuracy from FEWER transitions than from-scratch. That few-shot
gap is the faithful offline proxy for "adapts faster to a hidden game" -- the prior is trained on games the
held-out set is NOT in, so it measures transfer, not memorization.

Why this matters: the binding eval constraint is the per-level 5n action budget. A prior that cuts the
real probe actions needed to learn a new game's mechanics leaves more budget to solve efficiently = higher
score. We can never train on the hidden games; we CAN pretrain a transferable prior on the public 25.

CPU, offline, zero quota. Saves the prior to models/arc_dynamics_prior.pt.
Usage: .venv/bin/python scripts/arc_pretrain_prior.py [--epochs 10] [--fewshot 40] [--holdout g1,g2,...]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "python"))

import numpy as np

DEFAULT_HOLDOUT = ["cd82", "ka59", "sc25", "tn36", "lp85"]  # transfer-test games (NOT in the prior's training)


def _arg(argv, flag, default):
    return argv[argv.index(flag) + 1] if flag in argv else default


def _to_triples(transitions):
    from carnot.agentic.arc_live_ttt import action_key
    return [(t.grid, action_key(t.action, t.data), t.next_grid) for t in transitions]


def _acc_on_changing(model, transitions) -> tuple:
    """Exact-full-grid accuracy on the state-CHANGING transitions only (no-ops are trivial)."""
    chg = [t for t in transitions if not np.array_equal(t.grid, t.next_grid)]
    ok = sum(1 for t in chg if np.array_equal(np.asarray(model.predict(t.grid, _ak(t))), t.next_grid))
    return ok, len(chg)


def _ak(t):
    from carnot.agentic.arc_live_ttt import action_key
    return action_key(t.action, t.data)


def main(argv) -> int:
    import torch

    from carnot.agentic.arc_live_ttt import CNNDynamics
    from carnot.agentic.arc_transition_capture import TransitionCorpus

    epochs = int(_arg(argv, "--epochs", "10"))
    fewshot = int(_arg(argv, "--fewshot", "40"))
    holdout = (_arg(argv, "--holdout", "")).split(",") if _arg(argv, "--holdout", "") else DEFAULT_HOLDOUT

    corpus = TransitionCorpus()
    games = corpus.games()
    train_games = [g for g in games if g not in holdout]
    train_tr = []
    for g in train_games:
        train_tr += _to_triples(corpus.load(g))
    print(f"== pretrain prior on {len(train_games)} games ({len(train_tr)} transitions), holding out "
          f"{holdout} ==", flush=True)

    t0 = time.time()
    prior = CNNDynamics("prior", epochs=epochs).fit(train_tr, batch_size=256)
    state = prior.get_state()
    (REPO / "models").mkdir(exist_ok=True)
    torch.save(state, REPO / "models" / "arc_dynamics_prior.pt")
    print(f"  prior trained in {time.time()-t0:.0f}s -> models/arc_dynamics_prior.pt", flush=True)

    # few-shot transfer: for each held-out game, fit a learner on `fewshot` transitions of THAT game,
    # from-scratch vs warm-started-from-prior, and compare acc_on_changing on the remaining held-out trans.
    rows = []
    for g in holdout:
        tr = corpus.load(g)
        if len(tr) < fewshot + 10:
            rows.append({"game": g, "skip": f"too few ({len(tr)})"}); continue
        train, test = tr[:fewshot], tr[fewshot:]
        triples = _to_triples(train)
        scratch = CNNDynamics(g, epochs=epochs).fit(triples, batch_size=256)
        warm = CNNDynamics(g, epochs=epochs).fit(triples, batch_size=256, warm_state=state)
        s_ok, s_n = _acc_on_changing(scratch, test)
        w_ok, w_n = _acc_on_changing(warm, test)
        rows.append({"game": g, "fewshot": fewshot, "test_changing": s_n,
                     "scratch_acc": round(s_ok / max(1, s_n), 4), "warm_acc": round(w_ok / max(1, w_n), 4),
                     "warm_beats_scratch": (w_ok / max(1, w_n)) > (s_ok / max(1, s_n))})
        r = rows[-1]
        print(f"  {g:5} few-shot={fewshot}: scratch={r['scratch_acc']:.3f} warm={r['warm_acc']:.3f} "
              f"(changing n={s_n})  {'WARM WINS' if r['warm_beats_scratch'] else ''}", flush=True)

    scored = [r for r in rows if "warm_acc" in r]
    wins = sum(1 for r in scored if r["warm_beats_scratch"])
    mean_scratch = round(np.mean([r["scratch_acc"] for r in scored]), 4) if scored else 0.0
    mean_warm = round(np.mean([r["warm_acc"] for r in scored]), 4) if scored else 0.0
    out = {
        "experiment": "arc_pretrain_prior",
        "honest_verdict": f"complete_prior_transfer_warm_wins_{wins}_of_{len(scored)}",
        "epochs": epochs, "fewshot": fewshot, "holdout": holdout,
        "train_games": len(train_games), "train_transitions": len(train_tr),
        "mean_scratch_acc_on_changing": mean_scratch, "mean_warm_acc_on_changing": mean_warm,
        "warm_wins": wins, "held_out_games": len(scored), "per_game": rows,
        "prior_path": "models/arc_dynamics_prior.pt",
        "inference_substrate": "offline cross-game CNN dynamics pretraining + few-shot transfer; no LLM, no quota",
        "verifier_is_oracle": False,
    }
    (REPO / "results" / "arc_pretrain_prior.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\n  TRANSFER: warm-start beats scratch on {wins}/{len(scored)} held-out games; "
          f"mean acc_on_changing scratch={mean_scratch} -> warm={mean_warm}. -> {out['honest_verdict']}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

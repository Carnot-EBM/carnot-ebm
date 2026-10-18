"""M2-v2: does the object-level delta-DSL inducer fix the generalization bottleneck M2-v1a found?

Plan: docs/research-notes/arc-agi3-agent-research-plan.md (M2). M2-v1a showed the naive per-pixel
template inducer cannot generalize to unseen (state, action) even on deterministic games
(generalization energy ~0.95 on both Markov and hidden). This experiment runs BOTH inducers on the
identical collected transitions + identical key-disjoint split + identical grade_predictions verifier,
so the comparison is clean:

  - naive  = InducedWorldModel    (per-color relative-pixel template; the M2-v1a baseline)
  - dsl    = ObjectDeltaModel     (object-level translate/recolor rules; the M2-v2 inducer)

The win condition: the DSL's GENERALIZATION energy drops on the 14 grid-Markov games (object rules
generalize across positions where pixel templates can't), AND the consistency energy now SEPARATES
Markov (low = trustworthy model) from hidden-state (high = untrustworthy) using a model that actually
generalizes. That makes the energy verifier load-bearing: it certifies which induced world-models can
be trusted for planning. Offline, no LLM/GPU, no oracle.

  .venv/bin/python scripts/experiments/arc3_m2_dsl.py --budget 1200 --episodes 35
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "scripts" / "experiments"))
from carnot.agentic.arc_world_model_synth import InducedWorldModel, grade_predictions  # noqa: E402
from carnot.agentic.arc_world_model_dsl import ObjectDeltaModel  # noqa: E402
from arc3_m2_world_model import _collect, _key_disjoint_split, _random_split, _auroc  # noqa: E402


def run(games=None, budget=1200, episodes=35, seed=0, write=True):
    from arc_agi import Arcade
    from arc_agi.base import OperationMode
    from arcengine.enums import GameAction, GameState
    started = time.time()
    rng = random.Random(seed)
    arc = Arcade(arc_api_key="", operation_mode=OperationMode.OFFLINE,
                 environments_dir=str(REPO / "environment_files"))
    all_ids = sorted(getattr(e, "game_id", None) for e in arc.get_environments())
    sel = [g for g in all_ids if (not games or g.split("-")[0] in set(games))]
    probe = REPO / "results" / "arc3_determinism_probe.json"
    hidden = set(json.loads(probe.read_text()).get("hidden_state_games", [])) if probe.exists() else set()

    per_game = []
    for game in sel:
        short = game.split("-")[0]
        trans = _collect(arc, game, budget, episodes, rng, GameAction, GameState)
        tr_k, hd_k = _key_disjoint_split(trans, rng, frac=0.25)
        e_naive = grade_predictions(InducedWorldModel(short).fit(tr_k).predict, hd_k)
        e_dsl = grade_predictions(ObjectDeltaModel(short).fit(tr_k).predict, hd_k)
        is_hidden = short in hidden
        en, ed = e_naive["energy"], e_dsl["energy"]
        per_game.append({
            "game": short, "is_hidden_state_truth": is_hidden, "n_transitions": len(trans),
            "energy_gen_naive": en, "energy_gen_dsl": ed,
            "dyn_acc_naive": e_naive.get("dynamics_accuracy"), "dyn_acc_dsl": e_dsl.get("dynamics_accuracy"),
            "improvement": (round(en - ed, 4) if (en is not None and ed is not None) else None),
        })
        print(f"  {short:6s} naive={en} dsl={ed} improve={per_game[-1]['improvement']} hidden={is_hidden}",
              flush=True)

    rated = [g for g in per_game if g["energy_gen_dsl"] is not None and g["energy_gen_naive"] is not None]
    labels = [1 if g["is_hidden_state_truth"] else 0 for g in rated]
    auroc_naive = _auroc([g["energy_gen_naive"] for g in rated], labels)
    auroc_dsl = _auroc([g["energy_gen_dsl"] for g in rated], labels)
    markov = [g for g in rated if not g["is_hidden_state_truth"]]
    hid = [g for g in rated if g["is_hidden_state_truth"]]

    def _mean(gs, k):
        return round(sum(g[k] for g in gs) / len(gs), 4) if gs else None
    mk_naive, mk_dsl = _mean(markov, "energy_gen_naive"), _mean(markov, "energy_gen_dsl")
    hd_naive, hd_dsl = _mean(hid, "energy_gen_naive"), _mean(hid, "energy_gen_dsl")
    n_improved = sum(1 for g in rated if g["improvement"] and g["improvement"] > 0.05)
    best_dsl = sorted(markov, key=lambda g: g["energy_gen_dsl"])[:5]

    verdict = (f"complete: m2v2_dsl_inducer_genAUROC_naive{auroc_naive}_dsl{auroc_dsl}"
               f"_markovGenE_naive{mk_naive}_dsl{mk_dsl}_hiddenGenE_dsl{hd_dsl}"
               f"_n_improved{n_improved}of{len(rated)}")
    art = {
        "experiment": "arc3_m2_dsl", "title": "arc3_m2v2_object_dsl_vs_naive_inducer",
        "honest_verdict": verdict,
        "inference_substrate": "offline_arc_agi3_world_model_consistency_energy",
        "claim": ("Object-level delta-DSL inducer vs the M2-v1a naive pixel template, on identical "
                  "transitions/split/verifier. Win = DSL lowers GENERALIZATION energy on Markov games "
                  "and the consistency energy separates Markov (trustworthy) from hidden (not)."),
        "n_games": len(per_game),
        "generalization_auroc_naive": auroc_naive, "generalization_auroc_dsl": auroc_dsl,
        "mean_gen_energy_markov_naive": mk_naive, "mean_gen_energy_markov_dsl": mk_dsl,
        "mean_gen_energy_hidden_naive": hd_naive, "mean_gen_energy_hidden_dsl": hd_dsl,
        "n_games_dsl_improved_gt_0.05": n_improved,
        "best_modelled_markov_games_dsl": [{"game": g["game"], "energy": g["energy_gen_dsl"]} for g in best_dsl],
        "budget_per_game": budget, "episodes_per_game": episodes, "random_seed": seed,
        "no_llm_used": True, "no_gpu_used": True, "submitted_to_leaderboard": False,
        "duration_s": round(time.time() - started, 1), "per_game": per_game,
        "note": ("M2-v2 inducer upgrade. If DSL gen-energy on Markov games << naive (~0.95) AND the "
                 "energy now separates Markov from hidden (AUROC up), the verifier can certify "
                 "trustworthy models -> M2-v2b plans on the low-energy (trusted) games for first solves. "
                 "If the DSL also cannot generalize, the games need richer (object-relational / latent) "
                 "modeling -> honest finding caught before building the planner."),
    }
    if write:
        (REPO / "results" / "arc3_m2_dsl.json").write_text(json.dumps(art, indent=2, sort_keys=True) + "\n", "utf-8")
    print(f"\n-> {verdict}")
    print(f"   GENERALIZATION energy on MARKOV games: naive={mk_naive} -> dsl={mk_dsl} "
          f"(lower=better) | hidden dsl={hd_dsl}")
    print(f"   separation AUROC(gen energy -> hidden): naive={auroc_naive} -> dsl={auroc_dsl} | "
          f"DSL improved {n_improved}/{len(rated)} games")
    return art


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", default="", help="comma-separated short ids; empty = all 25")
    ap.add_argument("--budget", type=int, default=1200)
    ap.add_argument("--episodes", type=int, default=35)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    gl = [g.strip() for g in args.games.split(",") if g.strip()] or None
    run(games=gl, budget=args.budget, episodes=args.episodes, seed=args.seed)

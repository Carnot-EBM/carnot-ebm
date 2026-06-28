#!/usr/bin/env python3
"""PoE-World A/B: does the weighted-product world model PREDICT held-out transitions better than the
nulled max-vote ProductWorldModel and the single induced engine? (operator-directed 2026-06-28).

THE QUESTION (verifier-moat relevant, oracle-distinct): given the SAME pool of programmatic experts, does
combining them by a fitted/pruned WEIGHTED PRODUCT-OF-EXPERTS consensus (PoE-World, arXiv:2505.10819)
reproduce held-out transitions better -- changed-cell recall + exact accuracy -- than (a) the max-vote
ProductWorldModel (highest-trust-wins-each-cell; exp4749, NULLED), and (b) the single LLM-induced engine?

This is a FAST OFFLINE transition-prediction measurement -- NO live search, NO env rollout -- so it is not
subject to the live-search wall-clock that bounds first-win experiments. The PoE-vs-maxvote arm is even
LLM-FREE (both built from exact-delta experts harvested from transitions; only the combination differs),
isolating the combination contribution. The single-engine arm needs ONE induce LLM call (optional: skipped
with a recorded note if the GPU server is down -- the PoE-vs-maxvote core still reports).

Failed-Experiment Rerun Discipline: prior failure = exp4749 ProductWorldModel "dead/identity engine";
root cause = highest-trust-per-cell max-vote collapsed to ~identity; what is DIFFERENT = weighted-consensus
combination + held-out-fitted + pruned weights + no-change prior. retire_if_same_verdict: if PoE does NOT
beat max-vote on held-out cell_recall, this lever retires (do not re-propose).

3-way split per game: train (build experts) / val (fit+prune weights) / test (score). Metric = mean over
games of test changed-cell recall. solve_provenance n/a (no solve claim); verifier_is_oracle=False.

USAGE: arc_poe_world_model_ab.py [n_games] [n_transitions]
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

N_GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 4
N_TRANS = int(sys.argv[2]) if len(sys.argv) > 2 else 150
SEED = 20260628


def _mean(xs):
    xs = [float(x) for x in xs]
    return round(sum(xs) / len(xs), 4) if xs else 0.0


def _score(engine, heldout):
    from carnot.agentic.arc_executable_world_model import WorldModelVerifier

    vr = WorldModelVerifier(list(heldout))
    res = vr.score(engine)
    return {"cell_recall": round(float(res.cell_recall), 4), "accuracy": round(float(res.accuracy), 4)}


def _induce_single_engine(game, train, cell):
    """Optional single LLM-induced engine baseline. Returns (engine|None, note)."""
    try:
        from carnot.agentic.arc_executable_world_model import LocalGGUFProposer, load_engine

        proposer = LocalGGUFProposer(
            repo_substr="Qwen3.5-9B-MTP",
            model_path=os.environ.get("CARNOT_ARC_GGUF_PATH") or None,
            mtp=(os.environ.get("CARNOT_ARC_MTP", "1") != "0"),
            kv_quant="q8_0",
            no_think_prefix="/no_think\n",
            port=int(os.environ.get("CARNOT_IGE_LLM_PORT", "8919")),
        )
        if not proposer._healthy() and not proposer._ensure_server():
            return None, "llm_server_unreachable_single_engine_skipped"
        ok, _msg = proposer.induce(game, list(train), cell)
        if not ok:
            return None, f"induce_failed:{str(_msg)[:80]}"
        engine, _is_done = load_engine(game)
        return engine, "induced"
    except Exception as exc:
        return None, f"exception:{exc!r}"[:120]


def main() -> int:
    started = time.time()
    np.random.seed(SEED)
    from carnot.agentic import arc_poe_world_model as poe
    from carnot.agentic.arc_executable_world_model import ProductWorldModel, collect_transitions

    # offline scorer for the oracle-distinct energy reweight (best-effort; PoE works without it)
    energy_scorer = None
    try:
        from carnot.agentic.arc_world_model_trust_energy import default_s1_offpath_energy_scorer

        energy_scorer = default_s1_offpath_energy_scorer()
    except Exception:
        energy_scorer = None

    from carnot.experiment_4605_live_integration_scored_agent import _public_games

    games = _public_games(REPO)[:N_GAMES]
    per_game = []
    for game in games:
        try:
            trans, cell = collect_transitions(game, n=N_TRANS, seed=SEED)
        except Exception as exc:
            per_game.append({"game": game, "skipped": f"collect_failed:{exc!r}"[:100]})
            continue
        changed = [t for t in trans if not np.array_equal(t.grid, t.next_grid)]
        if len(changed) < 6:
            per_game.append({"game": game, "skipped": f"too_few_changed_transitions_{len(changed)}"})
            continue
        n = len(trans)
        a, b = int(n * 0.5), int(n * 0.75)
        train, val, test = trans[:a], trans[a:b], trans[b:]
        if not val:
            val = train
        if not test:
            test = val

        # SAME expert pool for PoE and max-vote; weights/trust fit on val (fit_poe_weights sets .trust too)
        pool = poe.build_expert_pool(train)
        weights = poe.fit_poe_weights(pool, val, energy_scorer=energy_scorer)
        poe_model = poe.PoEWorldModel(experts=pool, weights=weights)
        maxvote = ProductWorldModel(pool)  # reads .trust set by fit_poe_weights

        poe_s = _score(poe_model.engine, test)
        mv_s = _score(maxvote.engine, test)
        single_engine, single_note = _induce_single_engine(game, train, cell)
        single_s = _score(single_engine, test) if single_engine is not None else None

        per_game.append({
            "game": game,
            "n_transitions": n,
            "n_changed": len(changed),
            "n_experts": len(pool),
            "n_kept": int(sum(1 for w in weights if w > 0)),
            "poe": poe_s,
            "maxvote": mv_s,
            "single_engine": single_s,
            "single_engine_note": single_note,
        })

    scored = [g for g in per_game if "poe" in g]
    poe_recall = _mean([g["poe"]["cell_recall"] for g in scored])
    mv_recall = _mean([g["maxvote"]["cell_recall"] for g in scored])
    single_games = [g for g in scored if g.get("single_engine")]
    single_recall = _mean([g["single_engine"]["cell_recall"] for g in single_games]) if single_games else None

    poe_beats_maxvote = bool(scored) and poe_recall > mv_recall
    poe_beats_single = (single_recall is not None) and poe_recall > single_recall
    # per-game win counts (more robust than the mean on a few games)
    poe_gt_mv_games = sum(1 for g in scored if g["poe"]["cell_recall"] > g["maxvote"]["cell_recall"])
    poe_lt_mv_games = sum(1 for g in scored if g["poe"]["cell_recall"] < g["maxvote"]["cell_recall"])

    if not scored:
        verdict = "complete_poe_world_ab_no_scorable_games_inconclusive"
    elif poe_beats_maxvote and poe_gt_mv_games >= poe_lt_mv_games:
        single_clause = (
            f"_and_single_{single_recall}" if poe_beats_single else
            (f"_but_not_single_{single_recall}" if single_recall is not None else "_single_skipped")
        )
        verdict = (
            f"success_poe_world_beats_maxvote_cellrecall_{poe_recall}_vs_{mv_recall}"
            f"_wins_{poe_gt_mv_games}of{len(scored)}{single_clause}"
        )
    else:
        verdict = (
            f"complete_poe_world_no_lift_over_maxvote_cellrecall_{poe_recall}_vs_{mv_recall}"
            f"_wins_{poe_gt_mv_games}of{len(scored)}_retire_if_same"
        )

    art = {
        "experiment": "arc_poe_world_model_ab",
        "schema": "carnot.arc_poe_world_model_ab.v1",
        "honest_verdict": verdict,
        "question": (
            "does the weighted product-of-experts world model (PoE-World, arXiv:2505.10819) predict "
            "held-out transitions (changed-cell recall) better than the nulled max-vote ProductWorldModel "
            "(exp4749) and the single induced engine, given the SAME expert pool?"
        ),
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "verifier_is_oracle": False,
        "games": games,
        "n_transitions_target": N_TRANS,
        "poe_mean_cell_recall": poe_recall,
        "maxvote_mean_cell_recall": mv_recall,
        "single_engine_mean_cell_recall": single_recall,
        "poe_beats_maxvote": poe_beats_maxvote,
        "poe_beats_single_engine": poe_beats_single,
        "poe_gt_maxvote_games": poe_gt_mv_games,
        "poe_lt_maxvote_games": poe_lt_mv_games,
        "n_scored_games": len(scored),
        "per_game": per_game,
        "energy_reweighted": energy_scorer is not None,
        "prior_failures": [
            {
                "experiment_id": "exp4749",
                "verdict": "structured_productworldmodel_dead_identity_engine",
                "addressed_by": (
                    "weighted-product CONSENSUS combination (not highest-trust max-vote) + held-out-fitted "
                    "log-odds weights + pruning of <=chance experts + a no-change prior; decisive metric is "
                    "held-out changed-cell recall vs the max-vote baseline on the SAME expert pool."
                ),
                "retire_if_same_verdict": True,
            }
        ],
        "interpretation": (
            "poe_beats_maxvote=True -> the weighted-consensus combination extracts more predictive value "
            "from the same experts than max-vote: the genuine PoE-World contribution, oracle-distinct (the "
            "weights come from held-out accuracy + structural energy, never the win oracle). no_lift -> the "
            "combination rule is not the bottleneck; the experts themselves are too weak (consistent with "
            "the exp4749 null and the broader world-model induction ceiling) -> this lever retires."
        ),
        "cites_upstream": ["exp4749 (ProductWorldModel)", "exp4677 (programmatic experts)", "arXiv:2505.10819"],
        "model_specs": {"expert_source": "exact_delta+color_rewrite", "single_engine_generator": "unsloth/Qwen3.5-9B-MTP-GGUF"},
        "random_seed": SEED,
        "duration_s": round(time.time() - started, 2),
    }
    payload = dict(art)
    payload["reproducibility_checksum"] = ""
    art["reproducibility_checksum"] = "sha256:" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()
    out = REPO / "results" / "arc_poe_world_model_ab.json"
    out.write_text(json.dumps(art, indent=2) + "\n")
    print("\n=== VERDICT:", verdict)
    print(f"PoE cell_recall={poe_recall}  maxvote={mv_recall}  single={single_recall}  "
          f"(PoE>maxvote in {poe_gt_mv_games}/{len(scored)} games)")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

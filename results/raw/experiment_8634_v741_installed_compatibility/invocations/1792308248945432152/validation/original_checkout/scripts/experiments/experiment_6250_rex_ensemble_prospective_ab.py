#!/usr/bin/env python3
"""REQ-ARC-WMTE-6250: best-of-both REx ensemble, PROSPECTIVE A/B on NEW games.

WHAT THIS MEASURES. exp6248 found REx (UCB1+QBC) does not beat linear as a BLANKET
replacement (2 of 6 games) but the two arms fail on DIFFERENT games. A zero-cost
RETROSPECTIVE check on that same data found picking whichever arm has the higher VALID
score matches the HELD-optimal arm on 6 of 6 games -- but that check reused the exact
data that motivated it, so it cannot rule out the selection heuristic overfitting to
those 6 games. This experiment runs `run_rex_ensemble` LIVE on 4 games NONE of which
were in exp6248's roster (ft09, tr87, cn04, ar25, ka59, re86) -- a genuine held-out
test, not a re-read of old numbers.

OPERATOR AUTHORIZATION. 2026-08-11 operator directive "build a fresh prospective A/B for
the ensemble" -- the explicit, current operator decision this axis's standing hold
requires (same bar exp6248's own 2026-08-09 directive satisfied for the blanket-REx
test). This experiment does NOT re-propose the blanket-REx claim exp6248 retired: it
tests the DIFFERENT, NEW mechanism (per-game VALID-score selection between two already-
built arms) that exp6248's own retirement note flagged as unanswered.

PRE-REGISTERED GATE (weak, honest for n=4). The ensemble's pooled mean HELD
change_fidelity across the 4 games SHALL be >= the better of the pure-linear pooled mean
and the pure-rex pooled mean measured on THIS SAME roster (not exp6248's numbers -- a
fresh roster needs a fresh baseline, since neither arm's absolute performance transfers
across games). Secondary, reported not gated: per-game hit rate (does the VALID-chosen
arm match the HELD-optimal arm?), directly comparable to exp6248's retrospective 6/6.
n=4 is far below the project's n>=30 threshold for a percentage-point claim (CLAUDE.md
"Adversarial Artifact Verification + Sample-Size Rigor") -- this experiment reports a
directional signal, not a statistically powered one, and the artifact says so explicitly
regardless of outcome.

ISOLATION + INSTRUMENT PARITY WITH exp6248. Same CARNOT_ARC_E3_DIR private-scratch guard
(exp6247 incident), same CARNOT_ARC_REFACTOR_SHOW_ENGINE=1 forcing (exp5766 instrument
fix), same generator (gemma-4-31B-it-qat), same per-arm budget (4 LLM calls: 1 induce + 3
refinements) -- every methodology choice held constant so ANY difference in outcome is
attributable to the new roster, not to a confound between this script and exp6248's.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

if not os.environ.get("CARNOT_ARC_E3_DIR"):
    raise SystemExit(
        "set CARNOT_ARC_E3_DIR to a private scratch directory BEFORE launching this script -- "
        "it must never write to the shared results/arc_e3 store (exp6247 incident)."
    )
os.environ["CARNOT_ARC_REFACTOR_SHOW_ENGINE"] = "1"

from carnot.agentic import arc_executable_world_model as e3  # noqa: E402
from carnot.agentic import arc_rex_refinement as rex  # noqa: E402

OUT = REPO / "results" / "experiment_6250_rex_ensemble_prospective_ab.json"
CHECKPOINT = Path(
    os.environ.get("CARNOT_EXP6250_CHECKPOINT", "/tmp/carnot_exp6250_checkpoint.json")
)
# Deliberately disjoint from exp6248's roster (ft09, tr87, cn04, ar25, ka59, re86) -- a
# held-out test cannot reuse the games that motivated the hypothesis.
PRIOR_ROSTER = ("ft09", "tr87", "cn04", "ar25", "ka59", "re86")
ROSTER = ("dc22", "lp85", "sc25", "tu93")
N_COLLECT = 60
N_VALID = 10
N_HELD = 10
BUDGET = 4  # LLM calls per arm-cell: 1 induce + 3 refinements -- matches exp6248 exactly
SEED = 6250
GGUF_REPO_SUBSTR = "gemma-4-31B-it-qat"
SERVER_PORT = 8940
SHARED_MAX_TOKENS = 16384
INDUCE_TIMEOUT_S = 1500


def _load_checkpoint() -> dict:
    if CHECKPOINT.exists():
        return json.loads(CHECKPOINT.read_text())
    return {}


def _save_checkpoint(done: dict) -> None:
    CHECKPOINT.write_text(json.dumps(done, indent=2, default=str))


def _frame_hud_mask(game: str):
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_competition_agent import _compute_hud_mask_from_frame

    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    frame = env.reset()
    return _compute_hud_mask_from_frame(frame)


def _logical_hud_mask(game: str, cell: int):
    frame_mask = _frame_hud_mask(game)
    if frame_mask is None:
        return None
    return e3.logical_hud_mask(frame_mask, cell)


def _store_file(game: str) -> Path:
    return Path(os.environ["CARNOT_ARC_E3_DIR"]) / game / "world_model.py"


def _scorer(valid, hud_mask):
    """VALID-slice scorer for selection -- never sees HELD. Same shape as exp6248's."""

    def score_candidate(source: str) -> dict:
        try:
            engine = rex.load_engine_from_source(source, tag="score")
        except Exception as exc:  # noqa: BLE001
            return {
                "valid_fidelity": 0.0,
                "mismatches": [],
                "valid_accuracy": 0.0,
                "valid_n": 0,
                "valid_n_correct": 0,
                "load_error": repr(exc)[:160],
            }
        vr = e3.WorldModelVerifier(valid, hud_mask=hud_mask, hud_mask_enabled=True).score(
            engine, max_mismatch=len(valid)
        )
        return {
            "valid_fidelity": float(vr.change_fidelity),
            "mismatches": list(vr.mismatches),
            "valid_accuracy": float(vr.accuracy),
            "valid_n": int(vr.n),
            "valid_n_correct": int(vr.n_correct),
        }

    return score_candidate


def _make_verify_result(node: rex.RexNode, ordered_mismatches: list[dict]):
    return e3.VerifyResult(
        n=node.valid_n,
        n_correct=node.valid_n_correct,
        accuracy=node.valid_accuracy,
        mismatches=list(ordered_mismatches),
    )


def _score_held(source: str | None, held, hud_mask) -> dict:
    if not source:
        return {"held_change_fidelity": None}
    try:
        engine = rex.load_engine_from_source(source, tag="held")
    except Exception as exc:  # noqa: BLE001
        return {"held_change_fidelity": 0.0, "held_load_error": repr(exc)[:160]}
    vr = e3.WorldModelVerifier(held, hud_mask=hud_mask, hud_mask_enabled=True).score(engine)
    return {
        "held_change_fidelity": round(float(vr.change_fidelity), 4),
        "held_accuracy": round(float(vr.accuracy), 4),
        "held_cell_recall": round(float(vr.cell_recall), 4),
        "held_n_changing": int(vr.n_changing),
        "hud_mask_status": vr.hud_mask_status,
    }


def _run_game(prop, game: str, train, valid, held, hud_mask, cell: int) -> dict:
    store = _store_file(game)

    def read_store():
        return store.read_text() if store.exists() else None

    def write_store(text: str) -> None:
        store.parent.mkdir(parents=True, exist_ok=True)
        store.write_text(text)

    t0 = time.time()
    summary = rex.run_rex_ensemble(
        game,
        prop,
        train=train,
        valid=valid,
        cell=cell,
        budget=BUDGET,
        score_candidate=_scorer(valid, hud_mask),
        read_store_source=read_store,
        write_store_source=write_store,
        make_verify_result=_make_verify_result,
    )
    row: dict = {
        "game": game,
        "chosen_arm": summary["chosen_arm"],
        "chosen_final_valid_fidelity": summary["chosen_final_valid_fidelity"],
        "total_llm_calls": summary["total_llm_calls"],
        "wall_s": round(time.time() - t0, 1),
        "generator_healthy_after": bool(prop._healthy()),
    }
    linear_held = _score_held(summary["linear"].get("final_source"), held, hud_mask)
    rex_held = _score_held(summary["rex"].get("final_source"), held, hud_mask)
    row["linear_final_valid_fidelity"] = summary["linear"]["final_valid_fidelity"]
    row["rex_final_valid_fidelity"] = summary["rex"]["final_valid_fidelity"]
    row["linear_held_change_fidelity"] = linear_held.get("held_change_fidelity")
    row["rex_held_change_fidelity"] = rex_held.get("held_change_fidelity")
    row["ensemble_held_change_fidelity"] = _score_held(
        summary.get("chosen_final_source"), held, hud_mask
    ).get("held_change_fidelity")
    lh, rh = row["linear_held_change_fidelity"], row["rex_held_change_fidelity"]
    if lh is not None and rh is not None:
        held_optimal_arm = "rex" if rh > lh else ("linear" if lh > rh else "tie")
        row["held_optimal_arm"] = held_optimal_arm
        row["ensemble_matched_held_optimal"] = (
            held_optimal_arm == "tie" or summary["chosen_arm"] == held_optimal_arm
        )
    return row


def build_artifact() -> dict:
    t0 = time.time()
    os.environ["CARNOT_ARC_GENERATOR_CUDA_GPU"] = os.environ.get(
        "CARNOT_ARC_GENERATOR_CUDA_GPU", "1"
    )
    os.environ["CARNOT_ARC_GENERATOR_REQUIRE_CUDA"] = "1"

    from carnot.agentic.arc_executable_world_model import (
        GeneratorCudaRequiredError,
        LocalGGUFProposer,
    )

    prop = LocalGGUFProposer(
        repo_substr=GGUF_REPO_SUBSTR,
        port=SERVER_PORT,
        mtp=False,
        kv_quant="q8_0",
        max_tokens=SHARED_MAX_TOKENS,
        no_think_prefix="",
        timeout=INDUCE_TIMEOUT_S,
    )
    try:
        server_up = prop._ensure_server()
    except GeneratorCudaRequiredError as exc:
        return {"honest_verdict": f"complete_blocked_cuda_unavailable_{exc!r}"[:200]}
    if not server_up:
        return {"honest_verdict": "complete_blocked_cuda_server_failed_to_start"}

    done = _load_checkpoint()
    rows = list(done.get("rows", []))
    done_games = {r["game"] for r in rows if r.get("chosen_arm") is not None}
    rows = [r for r in rows if r["game"] in done_games]

    for game in ROSTER:
        if game in done_games:
            continue
        try:
            trans, cell = e3.collect_transitions(game, n=N_COLLECT, seed=SEED)
        except Exception as exc:  # noqa: BLE001
            rows.append({"game": game, "error": repr(exc)[:200]})
            done["rows"] = rows
            _save_checkpoint(done)
            continue
        train = trans[: -(N_VALID + N_HELD)]
        valid = trans[-(N_VALID + N_HELD) : -N_HELD]
        held = trans[-N_HELD:]
        hud_mask = _logical_hud_mask(game, cell)
        prop._ensure_server()
        try:
            row = _run_game(prop, game, train, valid, held, hud_mask, cell)
        except Exception as exc:  # noqa: BLE001
            row = {"game": game, "error": repr(exc)[:200]}
        rows.append(row)
        done["rows"] = rows
        _save_checkpoint(done)
        print(
            f"[exp6250] {game}: chosen={row.get('chosen_arm')} "
            f"linear_held={row.get('linear_held_change_fidelity')} "
            f"rex_held={row.get('rex_held_change_fidelity')} "
            f"ensemble_held={row.get('ensemble_held_change_fidelity')} "
            f"matched={row.get('ensemble_matched_held_optimal')}",
            flush=True,
        )

    comparable = [r for r in rows if r.get("ensemble_held_change_fidelity") is not None]
    n_comparable = len(comparable)
    linear_pooled = (
        round(sum(r["linear_held_change_fidelity"] for r in comparable) / n_comparable, 4)
        if n_comparable
        else None
    )
    rex_pooled = (
        round(sum(r["rex_held_change_fidelity"] for r in comparable) / n_comparable, 4)
        if n_comparable
        else None
    )
    ensemble_pooled = (
        round(sum(r["ensemble_held_change_fidelity"] for r in comparable) / n_comparable, 4)
        if n_comparable
        else None
    )
    n_matched = sum(1 for r in comparable if r.get("ensemble_matched_held_optimal"))
    best_pure_pooled = (
        max(linear_pooled, rex_pooled)
        if linear_pooled is not None and rex_pooled is not None
        else None
    )
    gate_met = (
        n_comparable == len(ROSTER)
        and ensemble_pooled is not None
        and best_pure_pooled is not None
        and ensemble_pooled >= best_pure_pooled
    )

    art = {
        "experiment": "experiment_6250_rex_ensemble_prospective_ab",
        "title": "REx ensemble prospective A/B: VALID-score arm selection on a held-out roster",
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "roster": list(ROSTER),
        "prior_roster_deliberately_excluded": list(PRIOR_ROSTER),
        "budget_llm_calls_per_arm_cell": BUDGET,
        "n_collect": N_COLLECT,
        "n_valid": N_VALID,
        "n_held": N_HELD,
        "per_game_results": rows,
        "n_games_comparable": n_comparable,
        "n_games_ensemble_matched_held_optimal": n_matched,
        "linear_pooled_mean_held": linear_pooled,
        "rex_pooled_mean_held": rex_pooled,
        "ensemble_pooled_mean_held": ensemble_pooled,
        "best_pure_arm_pooled_mean_held": best_pure_pooled,
        "gate_condition": "ensemble_pooled_mean_held >= best_pure_arm_pooled_mean_held",
        "gate_met": bool(gate_met),
        "sample_size_note": (
            "n=4 games, far below the project's n>=30 threshold for a percentage-point claim "
            "(CLAUDE.md Adversarial Artifact Verification + Sample-Size Rigor). This result is "
            "a directional signal on a genuinely held-out roster, not a statistically powered "
            "claim -- read n_games_ensemble_matched_held_optimal as a small, real data point "
            "alongside exp6248's retrospective 6/6, not as independent confirmation at scale."
        ),
        "surprising_result_acknowledgment": (
            "If ensemble_pooled_mean_held exceeds best_pure_arm_pooled_mean_held by a wide "
            "margin on n=4, that is preliminary, not headline-eligible until replicated on "
            "more games -- flagged per CLAUDE.md's cross-check-surprising-results rule."
        ),
        "engine_visible_refactor_prompt": True,
        "operator_authorization": (
            "2026-08-11 operator directive 'build a fresh prospective A/B for the ensemble' -- "
            "the explicit operator decision the standing refinement-axis hold requires, matching "
            "the bar exp6248's own 2026-08-09 directive satisfied."
        ),
        "prior_failures": [
            {
                "experiment_id": "exp6248",
                "verdict": "complete_rex_gate_not_met_2_of_6_games_pooled_delta_0.0602_variant_retired",
                "addressed_by": (
                    "exp6248 retired REx as a BLANKET replacement for linear -- this experiment "
                    "does not re-propose that claim. It tests a DIFFERENT mechanism (per-game "
                    "VALID-score selection between the two already-built arms) that exp6248's own "
                    "retirement note explicitly flagged as unanswered, on a roster disjoint from "
                    "the one that motivated the retrospective check."
                ),
            }
        ],
        "retire_if_same_verdict": False,
        "solve_provenance": "development_proxy",
        "arc_solve_claim": False,
        "verifier_is_oracle": False,
        "verifier_is_oracle_principle": (
            "held-out change_fidelity is an oracle-distinct graded dynamics score, not the "
            "executable win-condition oracle."
        ),
        "inference_substrate": "live_llm_inference",
        "model_specs": {
            "generator": "unsloth/gemma-4-31B-it-qat-GGUF",
            "quant": "UD-Q4_K_XL",
            "kv_cache_quant": "q8_0",
            "port": SERVER_PORT,
        },
        "random_seed": SEED,
    }
    if n_comparable == 0:
        art["honest_verdict"] = "complete_blocked_zero_comparable_games_not_a_lever_result"
    elif gate_met:
        art["honest_verdict"] = (
            f"complete_ensemble_gate_met_{n_matched}_of_{n_comparable}_matched_held_optimal_"
            f"ensemble_{ensemble_pooled}_vs_best_pure_{best_pure_pooled}"
        )
    else:
        art["honest_verdict"] = (
            f"complete_ensemble_gate_not_met_{n_matched}_of_{n_comparable}_matched_held_optimal_"
            f"ensemble_{ensemble_pooled}_vs_best_pure_{best_pure_pooled}"
        )
    art["duration_s"] = round(time.time() - t0, 3)
    payload = {k: v for k, v in art.items() if k != "duration_s"}
    art["reproducibility_checksum"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()
    return art


def main() -> int:
    art = build_artifact()
    OUT.write_text(json.dumps(art, indent=2, default=str) + "\n", encoding="utf-8")
    print("verdict:", art.get("honest_verdict"))
    print("wrote", OUT)
    if art.get("honest_verdict"):
        CHECKPOINT.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

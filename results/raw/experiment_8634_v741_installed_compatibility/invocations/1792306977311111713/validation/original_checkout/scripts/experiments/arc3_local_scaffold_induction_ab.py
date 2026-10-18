#!/usr/bin/env python3
"""Local-scaffold world-model induction A/B: does the ITERATIVE refactor loop, run by the
OFFLINE-LEGAL LOCAL model, lift the induced-world-model cell_recall above the ~0.05
near-identity wall?

WHY THIS EXPERIMENT EXISTS (the verify gate that motivated it, 2026-06-27)
-------------------------------------------------------------------------
The "e3 induction quality is the wall / cell_recall ~0.05" null
(results/arc_e3_induced_model_quality.json) was established on CODEX (gpt-5.5, a dev-only,
internet-REQUIRED proposer that is NOT legal in the air-gapped Kaggle eval) running the
full induce->verify->refactor loop. The OFFLINE-LEGAL LOCAL model (gemma-4-12B-it /
Qwen3.5-9B) running the SAME loop has NEVER had its induced-world-model cell_recall
measured (the local-proposer experiments exp4544/exp4557 recorded only refinement_rounds +
solve-rate, never WorldModelVerifier.cell_recall). That is the open gap this closes.

The frontier API is NOT an option for the challenge (no internet at eval), so the only
question that matters is: can the LOCAL model's executable-world-model induction clear the
wall? This is genuinely untried for the local model.

DESIGN (avoids the S2 degenerate-pool trap the operator caught twice)
---------------------------------------------------------------------
- Games: the e3 gap-1 set cn04, sc25, cd82, ka59 + ar25 as a POSITIVE CONTROL (codex got
  cell_recall 0.857 on ar25 => a known-inducible game; if the LOCAL model + harness also
  reach high cell_recall on ar25, a flat ~0 on the others is a REAL null, not a broken
  harness).
- ARM A (one-shot): LocalGGUFProposer.induce -> cell_recall on a HELD-OUT split (round 0).
- ARM B (iterative scaffold): N refactor rounds feeding the verifier's compact mismatch
  deltas back; re-score held-out each round; keep the trajectory + best.
- 2 seeds (0,1) so the transitions differ across runs (behavioral diversity guard).
- Metric: WorldModelVerifier(held_out).score(engine).cell_recall  (the granularity-matched
  gate metric; > 0.5 passes the live trust gate).
- Baseline: the documented codex per-game cell_recall (cn04 0.0146, sc25 0.0547, cd82 0.0,
  ka59 0.4628, ar25 0.857).

PRESERVES THE RECORD: induces into results/arc_e3_localind_ab/<game>__seed<k>/ via an
E3_DIR monkeypatch, so the codex-authored results/arc_e3/<game>/world_model.py baseline is
untouched (never-prune).

REUSES the already-running gemma-4-12B llama-server on :8919 (LocalGGUFProposer default
port + _ensure_server reuse), so no second model load and no GPU contention with the
conductor.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))

import numpy as np  # noqa: E402
from carnot.agentic import arc_executable_world_model as e3  # noqa: E402

GAMES = ["cn04", "sc25", "cd82", "ka59", "ar25"]  # ar25 = positive control
POSITIVE_CONTROL = "ar25"
SEEDS = [0, 1]
N_REFACTOR = 4
N_TRANS = 140
INDUCE_SPLIT = 100  # first 100 transitions induce; remainder held-out
GATE = 0.5
CODEX_BASELINE = {  # the documented codex cell_recall (results/arc_e3_induced_model_quality.json)
    "cn04": 0.0146,
    "sc25": 0.0547,
    "cd82": 0.0,
    "ka59": 0.4628,
    "ar25": 0.8567,
}

# Induce into a SEPARATE dir so the codex baseline (results/arc_e3/<game>/world_model.py) is preserved.
LOCAL_E3_DIR = REPO / "results" / "arc_e3_localind_ab"
LOCAL_E3_DIR.mkdir(parents=True, exist_ok=True)
e3.E3_DIR = LOCAL_E3_DIR  # monkeypatch: induce/refactor/load_engine all use this


def _cell_recall(engine, held) -> float | None:
    try:
        vr = e3.WorldModelVerifier(held).score(engine)
        return float(vr.cell_recall)
    except Exception:
        return None


def _score_for_refactor(engine, induce_trans):
    return e3.WorldModelVerifier(induce_trans).score(engine)


def run_game_seed(proposer, game: str, seed: int) -> dict:
    key = f"{game}__seed{seed}"
    out: dict = {"game": game, "seed": seed, "key": key}
    try:
        trans, cell = e3.collect_transitions(game, n=N_TRANS, seed=seed)
    except Exception as exc:
        out["error"] = f"collect_transitions: {repr(exc)[:160]}"
        return out
    n_changed = sum(1 for t in trans if not np.array_equal(t.grid, t.next_grid))
    out["n_trans"] = len(trans)
    out["n_changed"] = n_changed
    if len(trans) < INDUCE_SPLIT + 10:
        out["error"] = f"too few transitions ({len(trans)})"
        return out
    induce_trans = trans[:INDUCE_SPLIT]
    held = trans[INDUCE_SPLIT:]
    out["n_heldout"] = len(held)

    # ARM A: one-shot induce
    t0 = time.time()
    ok, msg = proposer.induce(key, induce_trans, cell)
    out["induce_ok"] = bool(ok)
    out["induce_msg"] = str(msg)[:160]
    if not ok:
        out["arm_a_cell_recall"] = None
        out["trajectory"] = []
        out["induce_s"] = round(time.time() - t0, 1)
        return out
    try:
        engine, _isdone = e3.load_engine(key)
    except Exception as exc:
        out["error"] = f"load_engine after induce: {repr(exc)[:160]}"
        return out
    cr0 = _cell_recall(engine, held)
    out["arm_a_cell_recall"] = cr0
    out["induce_s"] = round(time.time() - t0, 1)

    # ARM B: iterative refactor loop
    trajectory = [cr0]
    for r in range(1, N_REFACTOR + 1):
        try:
            vr = _score_for_refactor(engine, induce_trans)
            ok_r, _ = proposer.refactor(key, vr)
            if not ok_r:
                trajectory.append(None)
                continue
            engine, _isdone = e3.load_engine(key)
            trajectory.append(_cell_recall(engine, held))
        except Exception as exc:
            out.setdefault("refactor_errors", []).append(f"r{r}: {repr(exc)[:120]}")
            trajectory.append(None)
    out["trajectory"] = trajectory
    clean = [c for c in trajectory if c is not None]
    out["arm_b_best_cell_recall"] = max(clean) if clean else None
    return out


def main() -> int:
    started = time.time()
    # Preconditions
    pre = []
    try:
        from carnot.agentic import arc_solver_kit as kit

        kit.offline_arcade()
        pre.append({"resource": "offline_arcade", "available": True})
    except Exception as exc:
        print(json.dumps({"honest_verdict": "blocked_offline_arcade", "error": repr(exc)[:160]}))
        return 1
    proposer = e3.LocalGGUFProposer()  # default port 8919 + gemma-4-12B; reuses the running server
    if not proposer._ensure_server():
        art = {
            "experiment": "arc3_local_scaffold_induction_ab",
            "honest_verdict": "blocked_local_llama_server_unavailable",
            "inference_substrate": "live_llm_inference",
            "preconditions_checked": pre + [{"resource": "local_llama_server_8919", "available": False}],
            "duration_s": round(time.time() - started, 2),
        }
        (REPO / "results" / "arc3_local_scaffold_induction_ab.json").write_text(
            json.dumps(art, indent=2) + "\n"
        )
        print(json.dumps({"verdict": art["honest_verdict"]}))
        return 1
    pre.append({"resource": "local_llama_server_8919", "available": True, "model": proposer.repo_substr})

    rows = []
    for game in GAMES:
        for seed in SEEDS:
            row = run_game_seed(proposer, game, seed)
            rows.append(row)
            print(
                f"[{game} seed{seed}] armA={row.get('arm_a_cell_recall')} "
                f"armB_best={row.get('arm_b_best_cell_recall')} traj={row.get('trajectory')} "
                f"induce_ok={row.get('induce_ok')}",
                flush=True,
            )

    # Aggregate per game (best across seeds for ARM A and ARM B)
    per_game = {}
    for game in GAMES:
        grows = [r for r in rows if r["game"] == game]
        a_vals = [r.get("arm_a_cell_recall") for r in grows if r.get("arm_a_cell_recall") is not None]
        b_vals = [r.get("arm_b_best_cell_recall") for r in grows if r.get("arm_b_best_cell_recall") is not None]
        per_game[game] = {
            "codex_baseline_cell_recall": CODEX_BASELINE.get(game),
            "local_arm_a_best": max(a_vals) if a_vals else None,
            "local_arm_b_best": max(b_vals) if b_vals else None,
            "local_best": max(a_vals + b_vals) if (a_vals or b_vals) else None,
            "passes_gate": bool((max(a_vals + b_vals) if (a_vals or b_vals) else 0.0) >= GATE),
        }

    pc = per_game[POSITIVE_CONTROL]
    positive_control_passed = bool((pc["local_best"] or 0.0) >= GATE)
    # A lift = a non-control gap-1 game whose LOCAL induction materially beats its codex baseline AND clears the gate
    gap1 = [g for g in GAMES if g != POSITIVE_CONTROL]
    lifts = [
        g
        for g in gap1
        if per_game[g]["local_best"] is not None
        and per_game[g]["local_best"] >= GATE
        and per_game[g]["local_best"] > (CODEX_BASELINE.get(g, 0.0) + 0.1)
    ]
    refactor_helped = any(
        per_game[g]["local_arm_b_best"] is not None
        and per_game[g]["local_arm_a_best"] is not None
        and per_game[g]["local_arm_b_best"] > per_game[g]["local_arm_a_best"] + 0.05
        for g in GAMES
    )

    if not positive_control_passed:
        verdict = (
            "complete: local_scaffold_induction_inconclusive_positive_control_failed_"
            "harness_or_model_cannot_induce_even_ar25"
        )
    elif lifts:
        verdict = f"success: local_scaffold_induction_clears_wall_on_{'_'.join(lifts)}"
    else:
        verdict = (
            "complete: local_scaffold_induction_nulls_near_identity_local_model_confirms_"
            "008_wall_executable_python_induction_dead_offline"
        )

    artifact = {
        "experiment": "arc3_local_scaffold_induction_ab",
        "schema": "carnot.arc3_local_scaffold_induction_ab.v1",
        "honest_verdict": verdict,
        "inference_substrate": "live_llm_inference",
        "verifier_is_oracle": False,
        "solve_provenance": "development_proxy",
        "local_model": proposer.repo_substr,
        "games": GAMES,
        "positive_control_game": POSITIVE_CONTROL,
        "positive_control_passed": positive_control_passed,
        "positive_control_local_best": pc["local_best"],
        "n_refactor_rounds": N_REFACTOR,
        "seeds": SEEDS,
        "gate_threshold": GATE,
        "per_game": per_game,
        "lifts": lifts,
        "refactor_helped_any_game": refactor_helped,
        "rows": rows,
        "codex_baseline_provenance": (
            "results/arc_e3_induced_model_quality.json -- codex/gpt-5.5 (dev-only, "
            "internet-required) full refactor loop; the wall this measures the LOCAL model against."
        ),
        "methodology_note": (
            "Closes the open gap: the 0.05 near-identity null was codex (frontier); the offline-legal "
            "LOCAL model's induced-world-model cell_recall was never measured. ar25 is the positive "
            "control (codex 0.857). A flat near-identity on the gap-1 games WITH ar25 passing is a REAL "
            "null for the local executable-Python induction path (no internet at eval -> this is the "
            "model that matters). 2 seeds + held-out split guard against the S2 degenerate-pool trap."
        ),
        "preconditions_checked": pre,
        "random_seed": SEEDS[0],
        "random_seeds_used": SEEDS,
        "duration_s": round(time.time() - started, 2),
    }
    payload = dict(artifact)
    payload["reproducibility_checksum"] = ""
    artifact["reproducibility_checksum"] = "sha256:" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()
    (REPO / "results" / "arc3_local_scaffold_induction_ab.json").write_text(
        json.dumps(artifact, indent=2, default=str) + "\n"
    )
    print("\n=== VERDICT:", verdict)
    print("per_game:", json.dumps(per_game, indent=2))
    print("-> results/arc3_local_scaffold_induction_ab.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

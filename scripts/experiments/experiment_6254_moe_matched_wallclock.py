#!/usr/bin/env python3
"""REQ-ARC-WMTE-6254: MoE-many versus dense-few induction at MATCHED WALL-CLOCK.

THE QUESTION. The generator is pinned to dense gemma-4-31B, and that pin is settled: the
2026-07-28 head-to-head scored it 11-0-2 over the alternatives. But that head-to-head
measured PER-SAMPLE quality. A Mixture-of-Experts model activates only a fraction of its
weights per token, so it decodes far faster, and a faster model can draw more samples in
the same wall-clock. Best-of-N selection (REQ-ARC-WMTE-6251) turns extra samples into
extra quality. So a per-sample loss can still be a per-SECOND win, and nothing has
measured that. This experiment does.

THIS DOES NOT RE-OPEN THE GENERATOR PIN. It measures one number the pin decision never
had. Changing the pin needs a separate operator decision, and this artifact recommends
nothing on its own.

THE DESIGN, AND WHY MATCHED WALL-CLOCK IS THE ONLY FAIR FRAME.
  Phase 1: dense gemma-4-31B draws ONE sample per game. Record its wall-clock T.
  Phase 2: MoE Qwen3.6-35B-A3B draws samples for the SAME game until its cumulative
           wall-clock reaches T. Keep the best by VALID fidelity.
Comparing at matched sample count would beg the question, because the whole MoE claim is
about samples per second. Comparing at matched wall-clock asks the question the operator
actually has: given a fixed time budget, which model produces the better world model?

SEQUENTIAL SERVERS, NOT SIMULTANEOUS. The two models are 18-21 GB each in Q4. A local
RTX 3090 holds 24 GB, so they cannot both be resident. Phase 1 tears its server down
before Phase 2 starts. This costs one model load and removes any memory-pressure
confound between the arms.

ISOLATION. `CARNOT_ARC_E3_DIR` must be a private scratch directory (the exp6247
shared-store clobber incident). `CARNOT_ARC_REFACTOR_SHOW_ENGINE=1` is forced, matching
every sibling experiment in this family.
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

OUT = REPO / "results" / "experiment_6254_moe_matched_wallclock.json"
CHECKPOINT = Path(
    os.environ.get("CARNOT_EXP6254_CHECKPOINT", "/tmp/carnot_exp6254_checkpoint.json")
)
ROSTER = ("cn04", "s5i5", "ls20")
N_COLLECT = 60
N_VALID = 10
N_HELD = 10
SEED = 6254
DENSE_REPO_SUBSTR = "gemma-4-31B-it-qat"
MOE_REPO_SUBSTR = "Qwen3.6-35B-A3B"
DENSE_PORT = 8941
MOE_PORT = 8942
SHARED_MAX_TOKENS = 16384
INDUCE_TIMEOUT_S = 1500
MAX_MOE_SAMPLES = 8  # a hard stop, so a pathologically fast model cannot run forever


def _frame_hud_mask(game: str):
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_competition_agent import _compute_hud_mask_from_frame

    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    return _compute_hud_mask_from_frame(env.reset())


def _logical_hud_mask(game: str, cell: int):
    frame_mask = _frame_hud_mask(game)
    return None if frame_mask is None else e3.logical_hud_mask(frame_mask, cell)


def _scorer(valid, hud_mask):
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


def _make_verify_result(node, ordered):
    return e3.VerifyResult(
        n=node.valid_n,
        n_correct=node.valid_n_correct,
        accuracy=node.valid_accuracy,
        mismatches=list(ordered),
    )


def _score_held(source, held, hud_mask) -> float | None:
    if not source:
        return None
    try:
        engine = rex.load_engine_from_source(source, tag="held")
    except Exception:  # noqa: BLE001
        return 0.0
    vr = e3.WorldModelVerifier(held, hud_mask=hud_mask, hud_mask_enabled=True).score(engine)
    return round(float(vr.change_fidelity), 4)


def _make_proposer(repo_substr: str, port: int):
    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

    return LocalGGUFProposer(
        repo_substr=repo_substr,
        port=port,
        mtp=False,
        kv_quant="q8_0",
        max_tokens=SHARED_MAX_TOKENS,
        no_think_prefix="",
        timeout=INDUCE_TIMEOUT_S,
    )


def _one_sample(prop, game, train, valid, cell, hud_mask, store: Path) -> dict:
    def read_store():
        return store.read_text() if store.exists() else None

    def write_store(text: str) -> None:
        store.parent.mkdir(parents=True, exist_ok=True)
        store.write_text(text)

    t0 = time.time()
    summary = rex.run_rex(
        game,
        prop,
        train=train,
        valid=valid,
        cell=cell,
        budget=1,  # ONE induce call. This experiment varies COUNT, never refinement depth.
        score_candidate=_scorer(valid, hud_mask),
        read_store_source=read_store,
        write_store_source=write_store,
        make_verify_result=_make_verify_result,
        use_ucb1=False,
        use_qbc=False,
    )
    return {
        "final_source": summary["final_source"],
        "final_valid_fidelity": summary["final_valid_fidelity"],
        "llm_calls": summary["llm_calls"],
        "wall_s": round(time.time() - t0, 1),
    }


def _teardown(prop) -> None:
    """Stop this phase's server so the next model is not competing for VRAM."""
    try:
        prop._terminate_stale_proc("terminated: exp6254 phase complete, freeing the card")
    except Exception:  # noqa: BLE001
        pass


def build_artifact() -> dict:
    t0 = time.time()
    os.environ.setdefault("CARNOT_ARC_GENERATOR_CUDA_GPU", "1")
    os.environ["CARNOT_ARC_GENERATOR_REQUIRE_CUDA"] = "1"
    from carnot.agentic.arc_executable_world_model import GeneratorCudaRequiredError

    scratch = Path(os.environ["CARNOT_ARC_E3_DIR"])
    done = json.loads(CHECKPOINT.read_text()) if CHECKPOINT.exists() else {}
    rows = {r["game"]: r for r in done.get("rows", [])}

    prepared: dict[str, dict] = {}
    for game in ROSTER:
        trans, cell = e3.collect_transitions(game, n=N_COLLECT, seed=SEED)
        prepared[game] = {
            "train": trans[: -(N_VALID + N_HELD)],
            "valid": trans[-(N_VALID + N_HELD) : -N_HELD],
            "held": trans[-N_HELD:],
            "cell": cell,
            "hud_mask": _logical_hud_mask(game, cell),
        }

    # ---- Phase 1: dense, one sample per game -------------------------------------
    dense = _make_proposer(DENSE_REPO_SUBSTR, DENSE_PORT)
    try:
        if not dense._ensure_server():
            return {"honest_verdict": "complete_blocked_dense_server_failed_to_start"}
    except GeneratorCudaRequiredError as exc:
        return {"honest_verdict": f"complete_blocked_cuda_unavailable_{exc!r}"[:200]}
    for game in ROSTER:
        if game in rows and rows[game].get("dense_wall_s"):
            continue
        p = prepared[game]
        s = _one_sample(
            dense,
            game,
            p["train"],
            p["valid"],
            p["cell"],
            p["hud_mask"],
            scratch / f"dense_{game}" / "world_model.py",
        )
        rows.setdefault(game, {"game": game}).update(
            {
                "dense_wall_s": s["wall_s"],
                "dense_valid": s["final_valid_fidelity"],
                "dense_held": _score_held(s["final_source"], p["held"], p["hud_mask"]),
            }
        )
        done["rows"] = list(rows.values())
        CHECKPOINT.write_text(json.dumps(done, indent=2, default=str))
        print(
            f"[exp6254] dense {game}: held={rows[game]['dense_held']} wall={s['wall_s']}s",
            flush=True,
        )
    _teardown(dense)

    # ---- Phase 2: MoE, as many samples as fit in the dense arm's wall-clock -------
    moe = _make_proposer(MOE_REPO_SUBSTR, MOE_PORT)
    try:
        if not moe._ensure_server():
            for r in rows.values():
                r.setdefault("moe_blocked", "server failed to start")
            done["rows"] = list(rows.values())
            CHECKPOINT.write_text(json.dumps(done, indent=2, default=str))
    except GeneratorCudaRequiredError as exc:
        for r in rows.values():
            r.setdefault("moe_blocked", f"cuda unavailable {exc!r}"[:120])
    else:
        for game in ROSTER:
            if rows.get(game, {}).get("moe_n_samples"):
                continue
            p = prepared[game]
            budget_s = float(rows[game]["dense_wall_s"])
            arms, spent = {}, 0.0
            for i in range(MAX_MOE_SAMPLES):
                # CHECK BEFORE THE DRAW (corrected 2026-08-11 after adversarial review).
                # The first version checked AFTER, which always granted one unpaid final
                # sample -- and unboundedly so: a 100s dense budget plus one MoE sample
                # that hits the 1500s timeout handed the MoE arm 15x the budget. Matched
                # wall-clock is this experiment's stated premise, so it must be enforced,
                # not merely described. A first sample is always allowed; otherwise the
                # arm could return nothing and the comparison would be empty.
                if i > 0 and spent >= budget_s:
                    break
                s = _one_sample(
                    moe,
                    game,
                    p["train"],
                    p["valid"],
                    p["cell"],
                    p["hud_mask"],
                    scratch / f"moe_{game}_s{i}" / "world_model.py",
                )
                arms[f"sample{i}"] = s
                spent += s["wall_s"]
            chosen = rex.select_best_arm(game, arms)
            rows[game].update(
                {
                    "moe_n_samples": len(arms),
                    "moe_wall_s": round(spent, 1),
                    "moe_overspent_budget": bool(spent > budget_s),
                    "moe_budget_s": round(budget_s, 1),
                    "moe_best_valid": chosen["chosen_final_valid_fidelity"],
                    "moe_best_held": _score_held(
                        chosen.get("chosen_final_source"), p["held"], p["hud_mask"]
                    ),
                    "moe_per_sample_wall_s": [a["wall_s"] for a in arms.values()],
                }
            )
            done["rows"] = list(rows.values())
            CHECKPOINT.write_text(json.dumps(done, indent=2, default=str))
            print(
                f"[exp6254] moe {game}: n={len(arms)} best_held={rows[game]['moe_best_held']} "
                f"wall={spent:.0f}s of {budget_s:.0f}s budget",
                flush=True,
            )
        _teardown(moe)

    out_rows = list(rows.values())
    comparable = [
        r
        for r in out_rows
        if r.get("dense_held") is not None and r.get("moe_best_held") is not None
    ]
    n = len(comparable)
    n_moe_wins = sum(1 for r in comparable if r["moe_best_held"] > r["dense_held"])
    pooled_dense = round(sum(r["dense_held"] for r in comparable) / n, 4) if n else None
    pooled_moe = round(sum(r["moe_best_held"] for r in comparable) / n, 4) if n else None
    mean_samples = round(sum(r["moe_n_samples"] for r in comparable) / n, 2) if n else None
    moe_wins_pooled = bool(n and pooled_moe > pooled_dense)

    art = {
        "experiment": "experiment_6254_moe_matched_wallclock",
        "title": "MoE-many vs dense-few induction at matched wall-clock",
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "roster": list(ROSTER),
        "per_game_results": out_rows,
        "n_games_comparable": n,
        "n_games_moe_wins": n_moe_wins,
        "pooled_dense_held": pooled_dense,
        "pooled_moe_best_of_n_held": pooled_moe,
        "mean_moe_samples_within_dense_budget": mean_samples,
        "moe_wins_pooled": moe_wins_pooled,
        "gate_condition": (
            "reported, NOT gated. This experiment supplies one missing number for an "
            "operator decision. It does not by itself authorize a generator change."
        ),
        "does_not_reopen_generator_pin": (
            "the 2026-07-28 pin was decided on PER-SAMPLE quality (11-0-2). This measures "
            "quality per SECOND under best-of-N selection, which that decision never had. "
            "Any pin change needs a separate operator decision."
        ),
        "sample_size_note": (
            "3 games. Far below the project's n>=30 bar for a percentage-point claim. Read "
            "this as one data point for a decision, not as a result."
        ),
        "solve_provenance": "development_proxy",
        "arc_solve_claim": False,
        "verifier_is_oracle": False,
        "verifier_is_oracle_principle": (
            "held-out change_fidelity is an oracle-distinct graded dynamics score, not the "
            "executable win-condition oracle"
        ),
        "inference_substrate": "live_llm_inference",
        "model_specs": {
            "dense": "unsloth/gemma-4-31B-it-qat-GGUF (UD-Q4_K_XL)",
            "moe": "unsloth/Qwen3.6-35B-A3B-GGUF (UD-Q4_K_M)",
            "kv_cache_quant": "q8_0",
            "dense_port": DENSE_PORT,
            "moe_port": MOE_PORT,
        },
        "random_seed": SEED,
    }
    if n == 0:
        art["honest_verdict"] = "complete_blocked_zero_comparable_games_not_a_lever_result"
    else:
        art["honest_verdict"] = (
            f"complete_moe_matched_wallclock_measured_moe_wins_{n_moe_wins}_of_{n}_pooled_"
            f"moe_{pooled_moe}_vs_dense_{pooled_dense}_mean_samples_{mean_samples}"
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

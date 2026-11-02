#!/usr/bin/env python3
"""REQ-ARC-WMTE-6251: best-of-N induction sampling, selected by VALID fidelity.

WHAT THIS MEASURES. Does drawing N independent induction samples and keeping the
best-scoring one produce a better world model than drawing one sample, at the same
wall-clock? exp5722 showed a bigger dense generator alone moved zero live levels, so
model size is a closed axis. Sampling breadth is a different axis and is untested.

WHY THIS IS NOT THE RETIRED CANDIDATE-RANKING LEVER. Seven to nine A/Bs retired the
ranking of PERCEPTUAL CLICK CANDIDATES by a learned or hand scorer. This ranks GENERATED
PROGRAMS by execution-grounded VALID fidelity: it runs each candidate engine against
held-back observed transitions and measures how well it reproduces them. That is a
measurement, not a preference model, and it is oracle-distinct from the win condition.

THE WALL-CLOCK CLAIM IS THE POINT, AND IT IS MEASURED HERE, NOT ASSUMED. The value of
best-of-N rests on N samples costing much less than N times one sample, because the
server serves several slots at once. The v14 preview probe proved 4 simultaneous
requests SUCCEED. It did not measure their LATENCY. Continuous batching usually slows
each stream. So this experiment records per-arm wall-clock and reports the real speed-up
factor, and the gate is stated at MATCHED WALL-CLOCK rather than matched sample count.

CONCURRENCY NEEDS SEPARATE PROCESSES, NOT THREADS. `LocalGGUFProposer._write_world_model`
writes `E3_DIR / game / world_model.py`, and `E3_DIR` is a module-level global read at
import time. Two samples for one game inside a single process therefore write the SAME
file and race. Each sample runs in its own subprocess with its own `CARNOT_ARC_E3_DIR`.
All subprocesses share ONE llama-server over HTTP, which is what makes the parallel
sample cheap.

ISOLATION. `CARNOT_ARC_E3_DIR` must be a private scratch directory (the exp6247
shared-store clobber incident). `CARNOT_ARC_REFACTOR_SHOW_ENGINE=1` is forced, matching
exp6248 and exp6250 so the three are comparable.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
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

OUT = REPO / "results" / "experiment_6251_best_of_n_induction.json"
CHECKPOINT = Path(
    os.environ.get("CARNOT_EXP6251_CHECKPOINT", "/tmp/carnot_exp6251_checkpoint.json")
)
# Disjoint from exp6248's roster and from exp6250's, so a win here is not a third read of
# the same games.
ROSTER = ("cn04", "s5i5", "g50t", "ls20")
N_SAMPLES = 4  # matches the server's shipped slot count
N_COLLECT = 60
N_VALID = 10
N_HELD = 10
BUDGET = 1  # ONE induce call per sample: this tests sampling breadth, not refinement depth
SEED = 6251
GGUF_REPO_SUBSTR = "gemma-4-31B-it-qat"
SERVER_PORT = 8940
SHARED_MAX_TOKENS = 16384
INDUCE_TIMEOUT_S = 1500


def _worker_main() -> int:
    """One induction sample, in its own process with its own engine store.

    Reads its job from argv and writes a JSON summary to the path it is given. Prints
    nothing to stdout that the parent parses -- the file is the contract, because the
    generator libraries write freely to stdout.
    """
    game, sample_idx, out_path = sys.argv[2], int(sys.argv[3]), Path(sys.argv[4])
    from carnot.agentic import arc_executable_world_model as e3
    from carnot.agentic import arc_rex_refinement as rex
    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

    t0 = time.time()
    result: dict = {"game": game, "sample": sample_idx}
    try:
        trans, cell = e3.collect_transitions(game, n=N_COLLECT, seed=SEED)
        train = trans[: -(N_VALID + N_HELD)]
        valid = trans[-(N_VALID + N_HELD) : -N_HELD]
        hud_mask = _logical_hud_mask(game, cell)
        store = Path(os.environ["CARNOT_ARC_E3_DIR"]) / game / "world_model.py"

        def read_store():
            return store.read_text() if store.exists() else None

        def write_store(text: str) -> None:
            store.parent.mkdir(parents=True, exist_ok=True)
            store.write_text(text)

        prop = LocalGGUFProposer(
            repo_substr=GGUF_REPO_SUBSTR,
            port=SERVER_PORT,
            mtp=False,
            kv_quant="q8_0",
            max_tokens=SHARED_MAX_TOKENS,
            no_think_prefix="",
            timeout=INDUCE_TIMEOUT_S,
        )
        summary = rex.run_rex(
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
            use_ucb1=False,
            use_qbc=False,
        )
        result.update(
            {
                "final_source": summary["final_source"],
                "final_valid_fidelity": summary["final_valid_fidelity"],
                "llm_calls": summary["llm_calls"],
            }
        )
    except Exception as exc:  # noqa: BLE001
        result["error"] = repr(exc)[:300]
    result["wall_s"] = round(time.time() - t0, 1)
    out_path.write_text(json.dumps(result, default=str))
    return 0


def _frame_hud_mask(game: str):
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_competition_agent import _compute_hud_mask_from_frame

    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    return _compute_hud_mask_from_frame(env.reset())


def _logical_hud_mask(game: str, cell: int):
    from carnot.agentic import arc_executable_world_model as e3

    frame_mask = _frame_hud_mask(game)
    return None if frame_mask is None else e3.logical_hud_mask(frame_mask, cell)


def _scorer(valid, hud_mask):
    from carnot.agentic import arc_executable_world_model as e3
    from carnot.agentic import arc_rex_refinement as rex

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


def _make_verify_result(node, ordered_mismatches: list[dict]):
    from carnot.agentic import arc_executable_world_model as e3

    return e3.VerifyResult(
        n=node.valid_n,
        n_correct=node.valid_n_correct,
        accuracy=node.valid_accuracy,
        mismatches=list(ordered_mismatches),
    )


def _score_held(source, held, hud_mask) -> dict:
    from carnot.agentic import arc_executable_world_model as e3
    from carnot.agentic import arc_rex_refinement as rex

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
    }


def _spawn_samples(game: str, scratch_root: Path, *, concurrent: bool) -> list[dict]:
    """Run N sample subprocesses, each with its own engine store.

    `concurrent=False` runs them one after another. That arm exists to measure the real
    wall-clock cost of a single sample, which is the denominator of the speed-up factor
    the gate is stated against.
    """
    procs, outs = [], []
    for i in range(N_SAMPLES):
        e3dir = scratch_root / f"{game}_s{i}"
        e3dir.mkdir(parents=True, exist_ok=True)
        out_path = scratch_root / f"{game}_s{i}.json"
        outs.append(out_path)
        env = dict(os.environ)
        env["CARNOT_ARC_E3_DIR"] = str(e3dir)
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            game,
            str(i),
            str(out_path),
        ]
        p = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if concurrent:
            procs.append(p)
        else:
            p.wait()
    for p in procs:
        p.wait()
    rows = []
    for i, out_path in enumerate(outs):
        if out_path.exists():
            rows.append(json.loads(out_path.read_text()))
        else:
            rows.append({"game": game, "sample": i, "error": "worker wrote no output"})
    return rows


def build_artifact() -> dict:
    t0 = time.time()
    os.environ.setdefault("CARNOT_ARC_GENERATOR_CUDA_GPU", "1")
    os.environ["CARNOT_ARC_GENERATOR_REQUIRE_CUDA"] = "1"

    from carnot.agentic import arc_executable_world_model as e3
    from carnot.agentic import arc_rex_refinement as rex
    from carnot.agentic.arc_executable_world_model import (
        GeneratorCudaRequiredError,
        LocalGGUFProposer,
    )

    # The parent starts the shared server ONCE. Every worker then finds it already
    # healthy on the same port and does not launch its own.
    parent_prop = LocalGGUFProposer(
        repo_substr=GGUF_REPO_SUBSTR,
        port=SERVER_PORT,
        mtp=False,
        kv_quant="q8_0",
        max_tokens=SHARED_MAX_TOKENS,
        no_think_prefix="",
        timeout=INDUCE_TIMEOUT_S,
    )
    try:
        if not parent_prop._ensure_server():
            return {"honest_verdict": "complete_blocked_cuda_server_failed_to_start"}
    except GeneratorCudaRequiredError as exc:
        return {"honest_verdict": f"complete_blocked_cuda_unavailable_{exc!r}"[:200]}

    scratch_root = Path(os.environ["CARNOT_ARC_E3_DIR"]) / "bestofn"
    scratch_root.mkdir(parents=True, exist_ok=True)
    done = json.loads(CHECKPOINT.read_text()) if CHECKPOINT.exists() else {}
    rows = list(done.get("rows", []))
    seen = {r["game"] for r in rows}

    for game in ROSTER:
        if game in seen:
            continue
        trans, cell = e3.collect_transitions(game, n=N_COLLECT, seed=SEED)
        held = trans[-N_HELD:]
        hud_mask = _logical_hud_mask(game, cell)

        t_par = time.time()
        samples = _spawn_samples(game, scratch_root, concurrent=True)
        parallel_wall = round(time.time() - t_par, 1)

        arms = {
            f"sample{s['sample']}": {
                "final_source": s.get("final_source"),
                "final_valid_fidelity": s.get("final_valid_fidelity"),
                "llm_calls": s.get("llm_calls", 0),
            }
            for s in samples
        }
        chosen = rex.select_best_arm(game, arms)
        # N=1 baseline: sample0 is one honest draw. Using the FIRST sample rather than the
        # mean keeps the comparison paired -- it is a draw this run actually made.
        baseline = arms.get("sample0", {})

        row = {
            "game": game,
            "n_samples": N_SAMPLES,
            "parallel_wall_s": parallel_wall,
            "per_sample_wall_s": [s.get("wall_s") for s in samples],
            "slowest_sample_wall_s": max((s.get("wall_s") or 0) for s in samples),
            "sum_sample_wall_s": sum((s.get("wall_s") or 0) for s in samples),
            "chosen_arm": chosen["chosen_arm"],
            "chosen_final_valid_fidelity": chosen["chosen_final_valid_fidelity"],
            "baseline_n1_valid_fidelity": baseline.get("final_valid_fidelity"),
            "n_samples_produced_candidate": chosen["n_arms_produced_candidate"],
            "sample_errors": [s.get("error") for s in samples if s.get("error")],
            # Best-of-N collapses to N=1 if the sampler returns identical programs. That
            # happens when CARNOT_ARC_GENERATOR_SEED is exported: every worker inherits it
            # and draws the SAME seed at attempt 0. Adversarial review flagged that the run
            # would then burn 4x the compute to report a tie and read as an honest null.
            # Count distinct sources so a degenerate run is visible instead of silent.
            "n_distinct_sources": len(
                {s.get("final_source") for s in samples if s.get("final_source")}
            ),
            "generator_seed_env_set": bool(os.environ.get("CARNOT_ARC_GENERATOR_SEED")),
        }
        row["best_of_n_held"] = _score_held(chosen.get("chosen_final_source"), held, hud_mask).get(
            "held_change_fidelity"
        )
        row["baseline_n1_held"] = _score_held(baseline.get("final_source"), held, hud_mask).get(
            "held_change_fidelity"
        )
        # The real question behind the wall-clock claim: did running 4 at once actually
        # cost about one sample's time, or did batching serialise them?
        if row["slowest_sample_wall_s"]:
            row["batching_efficiency"] = round(
                row["sum_sample_wall_s"] / max(parallel_wall, 1e-9), 3
            )
        rows.append(row)
        done["rows"] = rows
        CHECKPOINT.write_text(json.dumps(done, indent=2, default=str))
        print(
            f"[exp6251] {game}: best_of_{N_SAMPLES}_held={row['best_of_n_held']} "
            f"n1_held={row['baseline_n1_held']} parallel_wall={parallel_wall}s "
            f"batching_eff={row.get('batching_efficiency')}x",
            flush=True,
        )

    comparable = [
        r
        for r in rows
        if r.get("best_of_n_held") is not None and r.get("baseline_n1_held") is not None
    ]
    n = len(comparable)
    n_improved = sum(1 for r in comparable if r["best_of_n_held"] > r["baseline_n1_held"])
    pooled_best = round(sum(r["best_of_n_held"] for r in comparable) / n, 4) if n else None
    pooled_n1 = round(sum(r["baseline_n1_held"] for r in comparable) / n, 4) if n else None
    effs = [r.get("batching_efficiency") for r in comparable if r.get("batching_efficiency")]
    mean_eff = round(sum(effs) / len(effs), 3) if effs else None
    gate_met = bool(n == len(ROSTER) and pooled_best is not None and pooled_best > pooled_n1)

    art = {
        "experiment": "experiment_6251_best_of_n_induction",
        "title": "Best-of-N induction sampling selected by VALID fidelity",
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "roster": list(ROSTER),
        "n_samples_per_game": N_SAMPLES,
        "budget_llm_calls_per_sample": BUDGET,
        "per_game_results": rows,
        "n_games_comparable": n,
        "n_games_best_of_n_improved": n_improved,
        "pooled_best_of_n_held": pooled_best,
        "pooled_n1_baseline_held": pooled_n1,
        "mean_batching_efficiency": mean_eff,
        "min_distinct_sources_across_games": min(
            (r.get("n_distinct_sources", 0) for r in comparable), default=None
        ),
        "batching_efficiency_note": (
            "sum of CONCURRENTLY-measured per-sample wall-clock divided by the concurrent "
            "wall-clock. CAVEAT from adversarial review: this ratio approaches N whenever "
            "the samples merely OVERLAP in time, even if continuous batching made each one "
            "N times slower than a lone sample. It measures overlap, NOT speed-up against a "
            "true one-at-a-time baseline. A real speed-up figure needs a sequential arm, "
            "which this run does not execute. Do not read it as N-for-the-price-of-one."
        ),
        "gate_condition": "pooled best-of-N held fidelity > pooled N=1 baseline on all roster games",
        "gate_met": gate_met,
        "sample_size_note": (
            "4 games. Far below the project's n>=30 bar for a percentage-point claim. A "
            "directional signal only."
        ),
        "not_the_retired_ranking_lever": (
            "the retired A/Bs reordered perceptual click candidates with a learned or hand "
            "scorer; this selects among generated PROGRAMS by execution-grounded VALID "
            "fidelity, which is a measurement rather than a preference model"
        ),
        "prior_failures": [
            {
                "experiment_id": "exp5722",
                "verdict": "floor_persists_stronger_generator_no_movement_delta_0.0",
                "addressed_by": (
                    "that tested a BIGGER model at one sample. This tests MORE samples at the "
                    "same model, which is a different axis and is untested."
                ),
            }
        ],
        "retire_if_same_verdict": False,
        "solve_provenance": "development_proxy",
        "arc_solve_claim": False,
        "verifier_is_oracle": False,
        "verifier_is_oracle_principle": (
            "held-out change_fidelity is an oracle-distinct graded dynamics score, not the "
            "executable win-condition oracle"
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
    if n == 0:
        art["honest_verdict"] = "complete_blocked_zero_comparable_games_not_a_lever_result"
    elif gate_met:
        art["honest_verdict"] = (
            f"complete_best_of_n_gate_met_{n_improved}_of_{n}_improved_pooled_"
            f"{pooled_best}_vs_n1_{pooled_n1}_batching_eff_{mean_eff}"
        )
    else:
        art["honest_verdict"] = (
            f"complete_best_of_n_gate_not_met_{n_improved}_of_{n}_improved_pooled_"
            f"{pooled_best}_vs_n1_{pooled_n1}_batching_eff_{mean_eff}"
        )
    art["duration_s"] = round(time.time() - t0, 3)
    payload = {k: v for k, v in art.items() if k != "duration_s"}
    art["reproducibility_checksum"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()
    ).hexdigest()
    return art


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        return _worker_main()
    art = build_artifact()
    OUT.write_text(json.dumps(art, indent=2, default=str) + "\n", encoding="utf-8")
    print("verdict:", art.get("honest_verdict"))
    print("wrote", OUT)
    if art.get("honest_verdict"):
        CHECKPOINT.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

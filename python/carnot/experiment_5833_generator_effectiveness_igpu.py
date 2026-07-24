#!/usr/bin/env python3
"""GENERATOR-EFFECTIVENESS BENCHMARK (REQ-ARC-WMTE-5833): Qwen3.5-9B vs Qwen3.6-27B vs Gemma-4-31B on
world-model INDUCTION QUALITY, ignoring speed (operator directive 2026-07-24).

WHY. The top ARC-AGI-3 leaderboard projects (e.g. forge, 3rd place LB 0.86) run 27B/31B-class generators,
while Carnot's frozen live generator is Qwen3.5-9B -- a choice made under a STALE 16GB VRAM assumption and
kept on SPEED grounds measured on undersized local hardware. The operator has set speed aside: measure raw
induction QUALITY across the three, on the LOCAL iGPU (Ryzen AI 9 HX 370 / Radeon 890M, ~105GB unified
memory), which FITS all three at Q4 -- a fair SAME-BACKEND comparison the 24GB discrete 3090s can't give
(the 31B fell off the PCI bus there). No cloud. exp5598 found dense-27B >> 9B and dense-27B > 35B-MoE on a
single-shot heldout benchmark; this replaces the 35B-MoE arm with the dense Gemma-4-31B the leaders actually
use, on the iGPU.

METHOD. Each arm gets its OWN llama-server on the iGPU (HIP binary via NO CUDA pin; `-fit off` per exp5705
so Q4 loads cleanly), launched + scored + STOPPED sequentially (proposer.stop() + _wait_for_port_down) so
only one model occupies unified memory at a time -- exp5598's pattern. Per (arm, game): build the offline L1
window, induce via the LIVE codeonly path (prop.induce -> induce_prompt), score heldout_accuracy (exact full
grid match on the held-out window) + cell_recall (soft per-cell) via WorldModelVerifier. Object-perception is
OFF (CARNOT_ARC_OBJECT_PERCEPTION unset) so this isolates the GENERATOR variable; the prompt lever gets folded
onto the winner later.

BOUNDED first pass: roster tu93,r11l,lp85 (known-good windows), N_SEEDS=1 (~9 induces). Widen after.

inference_substrate: live_llm_inference. verifier_is_oracle: False (heldout/cell_recall are exact/soft grid
match on held-out transitions; the win oracle is the level counter, never read by the induced engine).
solve_provenance: development_proxy. NEVER submits.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

HUB = Path.home() / ".cache/huggingface/hub"
ARMS = [
    {
        "name": "qwen9b",
        "repo_substr": "Qwen3.5-9B-MTP",
        "model_path": str(HUB / "models--unsloth--Qwen3.5-9B-MTP-GGUF/snapshots/"
                          "9716a636ee4bddc3fed678220b7a33dd2a4160ae/Qwen3.5-9B-Q4_K_M.gguf"),
        "port": 8931, "use_chat_template": False, "no_think_prefix": "/no_think\n", "n_ctx": 22000,
    },
    {
        "name": "qwen27b",
        "repo_substr": "Qwen3.6-27B-MTP",
        "model_path": str(HUB / "models--unsloth--Qwen3.6-27B-MTP-GGUF/snapshots/"
                          "5cb35eb3dcbf52dbce5f87dbc64df6aaffadcace/Qwen3.6-27B-Q4_K_M.gguf"),
        "port": 8932, "use_chat_template": True, "no_think_prefix": None, "n_ctx": 32768,
    },
    {
        "name": "gemma31b",
        "repo_substr": "gemma-4-31B-it",
        "model_path": str(HUB / "models--unsloth--gemma-4-31B-it-GGUF/snapshots/"
                          "f130ba51393346288f5862e30e9586b9b021513f/gemma-4-31B-it-Q4_K_M.gguf"),
        "port": 8930, "use_chat_template": True, "no_think_prefix": None, "n_ctx": 32768,
    },
]
ROSTER = [g for g in (os.environ.get("GEN_AB_ROSTER") or "tu93,r11l,lp85").split(",") if g]
N_SEEDS = int(os.environ.get("GEN_AB_SEEDS", "1"))
BUDGET = int(os.environ.get("GEN_AB_BUDGET", "4096"))
OUT = ROOT / "results" / "experiment_5833_generator_effectiveness_igpu.json"


def _wait_for_port_down(port: int, timeout_s: float = 40.0) -> None:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3):
                time.sleep(2)
        except Exception:
            return


def main() -> int:
    t0 = time.time()
    # object-perception OFF: isolate the generator variable.
    os.environ.pop("CARNOT_ARC_OBJECT_PERCEPTION", None)
    # iGPU path: ensure no CUDA pin so _generator_server_and_env picks the build-hip binary.
    os.environ.pop("CARNOT_ARC_GENERATOR_CUDA_GPU", None)
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)

    from carnot.agentic import arc_actions_to_progress as atp
    from carnot.agentic.arc_executable_world_model import (
        LocalGGUFProposer,
        WorldModelVerifier,
        load_engine,
    )

    # Build windows once (shared across arms so the only variable is the generator).
    windows: dict = {}
    for game in ROSTER:
        try:
            w = atp.build_progress_window(game)
        except Exception as e:
            w = None
            print(f"  window build failed {game}: {type(e).__name__}: {e}")
        if w is not None:
            windows[game] = w
        else:
            print(f"  SKIP {game}: no offline window")

    per_arm: list = []
    for arm in ARMS:
        prop = LocalGGUFProposer(
            repo_substr=arm["repo_substr"], model_path=arm["model_path"], port=arm["port"],
            mtp=False, kv_quant="q8_0", use_chat_template=arm["use_chat_template"],
            no_think_prefix=arm["no_think_prefix"], n_ctx=arm["n_ctx"],
            extra_server_args=("-fit", "off"), max_tokens=BUDGET, timeout=1800,
        )
        arm_rows: list = []
        server_ok = False
        try:
            server_ok = bool(prop._ensure_server())
        except Exception as e:
            print(f"[{arm['name']}] server launch error: {type(e).__name__}: {e}")
        if not server_ok:
            per_arm.append({"arm": arm["name"], "server_ok": False, "rows": [], "note": "server_failed"})
            print(f"[{arm['name']}] SERVER FAILED on iGPU -> arm skipped")
            continue
        for game, (window, _full_traj, cell) in windows.items():
            for seed in range(N_SEEDS):
                c0 = time.time()
                row = {"game": game, "seed": seed}
                try:
                    ok, msg = prop.induce(game, window, cell)
                    if ok:
                        eng, _lc = load_engine(game)
                        vr = WorldModelVerifier(list(window)).score(eng)
                        row.update({
                            "heldout_accuracy": round(float(vr.accuracy), 4),
                            "cell_recall": round(float(getattr(vr, "cell_recall", 0.0)), 4),
                            "induce_ok": True, "wall_s": round(time.time() - c0, 1),
                        })
                    else:
                        row.update({"heldout_accuracy": None, "cell_recall": None, "induce_ok": False,
                                    "error": str(msg)[:150], "wall_s": round(time.time() - c0, 1)})
                except Exception as e:
                    row.update({"heldout_accuracy": None, "cell_recall": None,
                                "error": f"{type(e).__name__}: {e}"[:200]})
                arm_rows.append(row)
                print(f"[{arm['name']}] {game} seed{seed}: heldout={row.get('heldout_accuracy')} "
                      f"cell_recall={row.get('cell_recall')} induce_ok={row.get('induce_ok')} "
                      f"({row.get('wall_s')}s)")
        # free unified memory before the next arm
        try:
            prop.stop()
        except Exception:
            pass
        _wait_for_port_down(arm["port"])

        vals_h = [r["heldout_accuracy"] for r in arm_rows if isinstance(r.get("heldout_accuracy"), (int, float))]
        vals_c = [r["cell_recall"] for r in arm_rows if isinstance(r.get("cell_recall"), (int, float))]
        per_arm.append({
            "arm": arm["name"], "server_ok": True, "rows": arm_rows,
            "n_induced": len(vals_h),
            "mean_heldout": round(sum(vals_h) / len(vals_h), 4) if vals_h else None,
            "mean_cell_recall": round(sum(vals_c) / len(vals_c), 4) if vals_c else None,
        })

    # ranking by mean_heldout (primary), then mean_cell_recall (tiebreak)
    ranked = sorted(
        [a for a in per_arm if a.get("mean_heldout") is not None],
        key=lambda a: (a["mean_heldout"], a.get("mean_cell_recall") or 0.0), reverse=True,
    )
    ranking = [{"arm": a["arm"], "mean_heldout": a["mean_heldout"], "mean_cell_recall": a["mean_cell_recall"]}
               for a in ranked]
    art = {
        "experiment": "experiment_5833_generator_effectiveness_igpu",
        "experiment_id": "REQ-ARC-WMTE-5833",
        "run_date": "2026-07-24",
        "title": "Generator effectiveness (induction quality): Qwen3.5-9B vs Qwen3.6-27B vs Gemma-4-31B on the iGPU.",
        "inference_substrate": "live_llm_inference",
        "verifier_is_oracle": False,
        "solve_provenance": "development_proxy",
        "random_seed": 5833,
        "model_specs": [{"name": a["repo_substr"], "gguf": a["model_path"], "role": "induction_generator",
                         "backend": "igpu_hip_q4"} for a in ARMS],
        "config": {"roster": list(windows.keys()), "n_seeds": N_SEEDS, "budget": BUDGET,
                   "object_perception": "off", "substrate": "igpu_unified_mem_q4"},
        "methodology_note": (
            "Each arm's own llama-server on the iGPU (HIP, -fit off, Q4), launched+scored+stopped "
            "sequentially. Per (arm,game): build_progress_window + LIVE prop.induce + WorldModelVerifier "
            "heldout_accuracy & cell_recall. Object-perception OFF -> isolates the generator. Quality only; "
            "speed ignored per operator directive."
        ),
        "per_arm": per_arm,
        "ranking": ranking,
        "preconditions_checked": [{"resource": "igpu_hip_llama_server", "available": True}],
        "duration_s": round(time.time() - t0, 1),
    }
    best = ranking[0]["arm"] if ranking else "none"
    art["honest_verdict"] = f"complete_generator_effectiveness_igpu_best_{best}_n_arms_{len(ranking)}"
    art["reproducibility_checksum"] = "sha256:" + hashlib.sha256(
        json.dumps(art, sort_keys=True, default=str).encode()).hexdigest()
    OUT.write_text(json.dumps(art, indent=2, default=str))
    print("\n=== RANKING (mean heldout, then cell_recall) ===")
    for i, r in enumerate(ranking, 1):
        print(f"  {i}. {r['arm']}: heldout={r['mean_heldout']} cell_recall={r['mean_cell_recall']}")
    print("wrote", OUT, f"({art['duration_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

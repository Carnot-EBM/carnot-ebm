#!/usr/bin/env python3
"""GENERATOR LIVE-LEVELS A/B (REQ-ARC-WMTE-5834): does the dense-27B generator bank MORE E3 levels than
the frozen 9B on the live path? (operator directive 2026-07-24)

WHY. exp5598 found the dense-27B a ~5x better world-model INDUCER offline (heldout 0.525 vs 9B 0.100), and
the 9B's only reason to stay live -- the 16GB Kaggle VRAM limit -- is stale now that the scored box is a
96GB RTX PRO 6000 Blackwell. BUT exp5722 ran a bigger (31B) inducer LIVE and it added 0 levels: the offline
induction gain did not translate, consistent with PERCEPTION (not induction) being the live binding
constraint. Offline heldout cannot settle the switch; only LIVE BANKED LEVELS can. This is that A/B.

METHOD. Two generator arms toggle ONLY the inducer model -- Qwen3.5-9B-MTP (current live) vs Qwen3.6-27B-MTP
(dense 27B) -- both on a 3090 via CUDA at Q4 (both fit 24GB: 9B ~6GB, 27B ~16GB; CUDA is fast+stable where
the iGPU HIP server wedges on the 27B). Sequential: launch arm server on the 3090, run each game through the
E3 agent's LIVE path (E3AgentPolicy + arc_leaderboard_eval.run_game, the OFFLINE dev twin of the scored
agent -- NEVER submits), record banked levels, stop server, next arm. Metric = levels banked per (arm,game).

GATE (pre-registered): dense-27B total levels > 9B total levels across the roster AND >=1 game where 27B>9B,
with NO game regressing (27B levels >= 9B levels for all). Clears -> the switch is LIVE-justified (recommend
rebuilding the submission stack on 27B-MTP). Else HONEST-NEGATIVE: the offline induction gain does NOT
convert to live levels (corroborates exp5722; perception is the bottleneck) -> keep 9B, target perception.

inference_substrate: live_llm_inference. verifier_is_oracle: False (levels come from the env level counter,
not read by the inducer). solve_provenance: development_proxy (offline dev twin; NEVER the scored submission).
GPU: 3090 GPU1 (outer-loop card) via CARNOT_ARC_GENERATOR_CUDA_GPU. iGPU untouched (lever#2 runs there).
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
sys.path.insert(0, str(ROOT / "scripts"))

HUB = Path.home() / ".cache/huggingface/hub"
CUDA_GPU = os.environ.get("GEN_LVL_CUDA_GPU", "1")  # 3090 GPU1 = outer-loop card (GPU0 = conductor)
ROSTER = [g for g in (os.environ.get("GEN_LVL_ROSTER") or "tu93,lp85,r11l").split(",") if g]
BUDGET = int(os.environ.get("GEN_LVL_BUDGET", "500"))         # E3 run_game action budget per game
EXPLORE_BUDGET = int(os.environ.get("GEN_LVL_EXPLORE", "40"))
INDUCE_TOKENS = int(os.environ.get("GEN_LVL_INDUCE_TOKENS", "2560"))
OUT = ROOT / "results" / "experiment_5834_generator_live_levels_ab.json"

ARMS = [
    {   # safe arm first: current live generator
        "name": "qwen9b", "repo_substr": "Qwen3.5-9B-MTP",
        "model_path": str(HUB / "models--unsloth--Qwen3.5-9B-MTP-GGUF/snapshots/"
                          "9716a636ee4bddc3fed678220b7a33dd2a4160ae/Qwen3.5-9B-Q4_K_M.gguf"),
        "port": 8951, "n_ctx": 22000,
    },
    {   # the candidate: dense 27B (exp5598's best inducer)
        "name": "qwen27b", "repo_substr": "Qwen3.6-27B-MTP",
        "model_path": str(HUB / "models--unsloth--Qwen3.6-27B-MTP-GGUF/snapshots/"
                          "5cb35eb3dcbf52dbce5f87dbc64df6aaffadcace/Qwen3.6-27B-Q4_K_M.gguf"),
        "port": 8952, "n_ctx": 32768,
    },
]


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
    # CUDA path on the 3090 (GPU1). Set BEFORE importing the proposer so _generator_server_and_env picks the
    # build-cuda binary on the right GPU. Object-perception OFF -> isolate the generator variable.
    os.environ["CARNOT_ARC_GENERATOR_CUDA_GPU"] = CUDA_GPU
    os.environ.pop("CARNOT_ARC_OBJECT_PERCEPTION", None)

    import arc_leaderboard_eval as lb
    from carnot.agentic.arc_competition_agent import E3AgentPolicy
    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

    per_arm: list = []
    for arm in ARMS:
        # mtp=False (uniform): the CUDA build crashes on --spec-type draft-mtp for these -MTP- GGUFs
        # (verified 2026-07-24: 27B loads clean on CUDA without it, crashes with it). MTP is
        # speculative decoding -> output-preserving, so dropping it changes ONLY speed, not the
        # induced world model or banked levels. Fair for a quality/levels A/B.
        prop = LocalGGUFProposer(
            repo_substr=arm["repo_substr"], model_path=arm["model_path"], port=arm["port"],
            mtp=False, kv_quant="q8_0", n_ctx=arm["n_ctx"], no_think_prefix="/no_think\n",
            max_tokens=INDUCE_TOKENS, timeout=1800, extra_server_args=("-fit", "off"),
        )
        server_ok = False
        try:
            server_ok = bool(prop._ensure_server())
        except Exception as e:
            print(f"[{arm['name']}] server launch error: {type(e).__name__}: {e}", flush=True)
        if not server_ok:
            per_arm.append({"arm": arm["name"], "server_ok": False, "rows": [], "note": "server_failed_on_3090_cuda"})
            print(f"[{arm['name']}] SERVER FAILED on 3090 CUDA -> arm skipped (no fabrication)", flush=True)
            continue
        rows: list = []
        for game in ROSTER:
            c0 = time.time()
            row = {"game": game}
            try:
                policy = E3AgentPolicy(game, proposer=prop, explore_budget=EXPLORE_BUDGET)
                r = lb.run_game(game, policy, budget=BUDGET)
                row.update({"levels": int(r.get("levels", 0)), "ok": True, "wall_s": round(time.time() - c0, 1)})
            except Exception as e:
                row.update({"levels": None, "ok": False, "error": f"{type(e).__name__}: {e}"[:200],
                            "wall_s": round(time.time() - c0, 1)})
            rows.append(row)
            print(f"[{arm['name']}] {game}: levels={row.get('levels')} ok={row.get('ok')} ({row.get('wall_s')}s)",
                  flush=True)
        try:
            prop.stop()
        except Exception:
            pass
        _wait_for_port_down(arm["port"])
        lvls = [r["levels"] for r in rows if isinstance(r.get("levels"), int)]
        per_arm.append({
            "arm": arm["name"], "server_ok": True, "rows": rows,
            "n_games": len(lvls), "total_levels": sum(lvls) if lvls else None,
        })

    # Gate: 27B total > 9B total AND >=1 game 27B>9B AND no game regresses.
    def _by(name):
        return next((a for a in per_arm if a["arm"] == name), None)
    a9, a27 = _by("qwen9b"), _by("qwen27b")
    per_game_delta = []
    gate_clears = False
    improved = regressed = 0
    if a9 and a27 and a9.get("server_ok") and a27.get("server_ok"):
        l9 = {r["game"]: r.get("levels") for r in a9["rows"]}
        l27 = {r["game"]: r.get("levels") for r in a27["rows"]}
        for g in ROSTER:
            v9, v27 = l9.get(g), l27.get(g)
            d = (v27 - v9) if isinstance(v9, int) and isinstance(v27, int) else None
            per_game_delta.append({"game": g, "qwen9b_levels": v9, "qwen27b_levels": v27, "delta": d})
            if isinstance(d, int):
                if d > 0:
                    improved += 1
                elif d < 0:
                    regressed += 1
        t9, t27 = a9.get("total_levels"), a27.get("total_levels")
        gate_clears = bool(
            isinstance(t9, int) and isinstance(t27, int) and t27 > t9 and improved >= 1 and regressed == 0
        )

    art = {
        "experiment": "experiment_5834_generator_live_levels_ab",
        "experiment_id": "REQ-ARC-WMTE-5834",
        "run_date": "2026-07-24",
        "title": "Generator live-levels A/B: Qwen3.5-9B vs dense Qwen3.6-27B, banked E3 levels on the 3090.",
        "inference_substrate": "live_llm_inference",
        "verifier_is_oracle": False,
        "solve_provenance": "development_proxy",
        "random_seed": 5834,
        "model_specs": [{"name": a["repo_substr"], "gguf": a["model_path"], "role": "e3_induction_generator",
                         "backend": f"cuda_3090_gpu{CUDA_GPU}_q4"} for a in ARMS],
        "config": {"roster": ROSTER, "budget": BUDGET, "explore_budget": EXPLORE_BUDGET,
                   "induce_tokens": INDUCE_TOKENS, "cuda_gpu": CUDA_GPU, "object_perception": "off", "mtp": False},
        "methodology_note": (
            "Two arms toggle ONLY the inducer model (9B vs dense-27B), both on the 3090 GPU%s via CUDA Q4, "
            "mtp=False (uniform; the CUDA build crashes on --spec-type draft-mtp -- MTP is output-preserving "
            "speculative decoding so this changes only speed, not induced quality). Sequential launch+run+stop. "
            "Per game: E3AgentPolicy + arc_leaderboard_eval.run_game "
            "(offline dev twin of the scored path -- NEVER submits); metric = banked levels. Settles the "
            "switch on the LIVE metric that exp5722 said the offline heldout gain failed to move." % CUDA_GPU
        ),
        "per_arm": per_arm,
        "per_game_delta": per_game_delta,
        "games_improved_27b": improved,
        "games_regressed_27b": regressed,
        "gate_clears": gate_clears,
        "preconditions_checked": [{"resource": f"cuda_3090_gpu{CUDA_GPU}", "available": True}],
        "duration_s": round(time.time() - t0, 1),
    }
    t9 = a9.get("total_levels") if a9 else None
    t27 = a27.get("total_levels") if a27 else None
    art["honest_verdict"] = (
        f"complete_generator_live_levels_ab_qwen9b_total_{t9}_qwen27b_total_{t27}"
        f"_improved_{improved}_regressed_{regressed}_gate_clears_{gate_clears}"
    )
    art["reproducibility_checksum"] = "sha256:" + hashlib.sha256(
        json.dumps(art, sort_keys=True, default=str).encode()).hexdigest()
    OUT.write_text(json.dumps(art, indent=2, default=str))
    print(f"\n=== LIVE LEVELS A/B: 9B total={t9} vs 27B total={t27} | improved={improved} regressed={regressed} "
          f"| GATE_CLEARS={gate_clears} ===", flush=True)
    print("wrote", OUT, f"({art['duration_s']}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

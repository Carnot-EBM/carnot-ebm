#!/usr/bin/env python3
"""LEVER #1 A/B (REQ-ARC-WMTE-5830): does object-structured perception in the inducer prompt lift
world-model induction heldout_accuracy?

Two arms differing ONLY by CARNOT_ARC_OBJECT_PERCEPTION (0 vs 1). Same generator (the live
Qwen3.5-9B-MTP server), same window per game, same seed/budget -> the ONLY scientific variable is the
object block. Reuses exp5726.run_reason_cell_budget VERBATIM (the exact induce+score cell that produced
the "29/37 inductions at heldout 0.0" null) and atp.build_progress_window. Metric = heldout_accuracy
(WorldModelVerifier exact-full-grid-match on the held-out window tail).

GATE (pre-registered, matching the scoping plan): paired mean heldout delta (on - off) > 0 across games
AND >= 1 game newly crossing the 0.5 live trust gate under ON that did not under OFF. Else HONEST-NEGATIVE
(record; do NOT re-propose bigger object serializations without a new mechanism).

inference_substrate: live_llm_inference. verifier_is_oracle: False (heldout = exact-match on held-out
transitions; the win oracle is the level counter, never read by the induced engine). solve_provenance:
development_proxy (measurement; the flag stays default-off unless the operator graduates it).
NEVER submits. GPU: reuses the running server on --port (default 8921); if unhealthy -> blocked_*.
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

import numpy as np  # noqa: E402

# Bounded roster (games with a valid offline L1 window; build_progress_window skips any without one).
DEFAULT_ROSTER = ["ls20", "tu93", "r11l", "lp85", "sc25", "cd82", "sk48", "sp80"]
ROSTER = [g for g in (os.environ.get("L1AB_ROSTER") or ",".join(DEFAULT_ROSTER)).split(",") if g]
PORT = int(os.environ.get("L1AB_PORT", "8921"))
BUDGET = int(os.environ.get("L1AB_BUDGET", "4096"))
TRIAL = 0
TRUST_GATE = 0.5
OUT = ROOT / "results" / "experiment_5831_object_perception_induction_ab.json"


def _server_healthy(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def main() -> int:
    t0 = time.time()
    # PRECONDITIONS (Pre-Launch Preconditions Discipline): a healthy generator must be reachable; else
    # blocked_* (no fabrication).
    pre = [{"resource": f"llama_server_port_{PORT}", "available": _server_healthy(PORT)}]
    if not pre[0]["available"]:
        art = {
            "experiment": "experiment_5831_object_perception_induction_ab",
            "experiment_id": "REQ-ARC-WMTE-5830",
            "honest_verdict": f"complete: blocked_no_generator_server_on_port_{PORT}",
            "inference_substrate": "live_llm_inference",
            "preconditions_checked": pre,
            "duration_s": round(time.time() - t0, 2),
        }
        OUT.write_text(json.dumps(art, indent=2))
        print("BLOCKED: no healthy generator on port", PORT, "->", OUT)
        return 0

    from carnot.agentic import arc_actions_to_progress as atp
    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer
    from carnot.experiment_5726_thinkingcap_16k_dualgpu_reason_ab import run_reason_cell_budget

    prop = LocalGGUFProposer(
        repo_substr="Qwen3.5-9B-MTP", port=PORT, mtp=True, kv_quant="q8_0",
        no_think_prefix="/no_think\n", max_tokens=BUDGET, timeout=600,
    )

    # Build one window per game up front (shared by both arms so the object variable is isolated).
    windows: dict[str, tuple] = {}
    for game in ROSTER:
        try:
            w = atp.build_progress_window(game)
        except Exception as e:
            w = None
            print(f"  window build failed {game}: {type(e).__name__}: {e}")
        if w is not None:
            windows[game] = w
        else:
            print(f"  SKIP {game}: no offline L1 window")

    per_game = []
    for game, (window, full_traj, cell) in windows.items():
        row: dict = {"game": game}
        for arm, flag in (("off", "0"), ("on", "1")):
            os.environ["CARNOT_ARC_OBJECT_PERCEPTION"] = flag  # set BEFORE induce (read in induce_prompt)
            try:
                r = run_reason_cell_budget(
                    game, prop, trial=TRIAL, window=window, full_traj=full_traj, cell=cell, budget=BUDGET
                )
                row[arm] = {
                    "heldout_accuracy": r.get("heldout_accuracy"),
                    "cell_recall": r.get("cell_recall"),
                    "goal_predicate_accuracy": r.get("goal_predicate_accuracy"),
                    "induce_ok": r.get("induce_ok"),
                    "wall_s": r.get("wall_s"),
                }
            except Exception as e:
                row[arm] = {"heldout_accuracy": None, "error": f"{type(e).__name__}: {e}"[:200]}
        os.environ.pop("CARNOT_ARC_OBJECT_PERCEPTION", None)
        ho_off = row.get("off", {}).get("heldout_accuracy")
        ho_on = row.get("on", {}).get("heldout_accuracy")
        row["heldout_delta"] = (
            round(float(ho_on) - float(ho_off), 6)
            if isinstance(ho_off, (int, float)) and isinstance(ho_on, (int, float))
            else None
        )
        per_game.append(row)
        print(f"[{game}] heldout off={ho_off} on={ho_on} delta={row['heldout_delta']}")

    # Gate
    paired = [
        (r["off"]["heldout_accuracy"], r["on"]["heldout_accuracy"])
        for r in per_game
        if isinstance(r.get("off", {}).get("heldout_accuracy"), (int, float))
        and isinstance(r.get("on", {}).get("heldout_accuracy"), (int, float))
    ]
    n = len(paired)
    mean_off = sum(o for o, _ in paired) / n if n else None
    mean_on = sum(v for _, v in paired) / n if n else None
    mean_delta = round(mean_on - mean_off, 6) if n else None
    new_trust_crossings = [
        r["game"]
        for r in per_game
        if isinstance(r.get("on", {}).get("heldout_accuracy"), (int, float))
        and isinstance(r.get("off", {}).get("heldout_accuracy"), (int, float))
        and r["on"]["heldout_accuracy"] >= TRUST_GATE
        and r["off"]["heldout_accuracy"] < TRUST_GATE
    ]
    gate_clears = bool(n and mean_delta is not None and mean_delta > 0 and len(new_trust_crossings) >= 1)
    seed = 5831
    art = {
        "experiment": "experiment_5831_object_perception_induction_ab",
        "experiment_id": "REQ-ARC-WMTE-5830",
        "run_date": "2026-07-24",
        "title": "Lever #1 A/B: object-structured perception in the inducer prompt vs raw grid; heldout_accuracy.",
        "inference_substrate": "live_llm_inference",
        "verifier_is_oracle": False,
        "solve_provenance": "development_proxy",
        "random_seed": seed,
        "model_specs": [{"name": "Qwen3.5-9B-MTP-GGUF", "port": PORT, "role": "induction_generator"}],
        "config": {"roster": list(windows.keys()), "budget": BUDGET, "trial": TRIAL, "trust_gate": TRUST_GATE},
        "methodology_note": (
            "Two arms toggle ONLY CARNOT_ARC_OBJECT_PERCEPTION; same live Qwen3.5-9B-MTP server, same "
            "window/seed/budget per game. run_reason_cell_budget (exp5726) reused verbatim. heldout_accuracy "
            "= WorldModelVerifier exact-full-grid-match on the held-out window tail."
        ),
        "per_game": per_game,
        "n_paired": n,
        "mean_heldout_off": None if mean_off is None else round(mean_off, 6),
        "mean_heldout_on": None if mean_on is None else round(mean_on, 6),
        "mean_heldout_delta": mean_delta,
        "new_trust_gate_crossings": new_trust_crossings,
        "gate_clears": gate_clears,
        "preconditions_checked": pre,
        "duration_s": round(time.time() - t0, 1),
    }
    art["honest_verdict"] = (
        f"complete_object_perception_ab_mean_delta_{mean_delta}_new_trust_{len(new_trust_crossings)}"
        f"_gate_clears_{gate_clears}"
    )
    art["reproducibility_checksum"] = "sha256:" + hashlib.sha256(
        json.dumps(art, sort_keys=True, default=str).encode()
    ).hexdigest()
    OUT.write_text(json.dumps(art, indent=2, default=str))
    print(
        f"\nAGGREGATE n={n} mean_heldout off={art['mean_heldout_off']} on={art['mean_heldout_on']} "
        f"delta={mean_delta} | new_trust_crossings={new_trust_crossings} | GATE_CLEARS={gate_clears}"
    )
    print("wrote", OUT, f"({art['duration_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

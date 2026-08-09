#!/usr/bin/env python3
"""REQ-ARC-WMTE-6246: the leave-one-game-out held-out change-fidelity A/B Phase 3a's own gate
requires (REQ-ARC-WMTE-6241 built and unit-tested `SUBMITTED_INDUCE_PROMPT_ENRICHMENT_ENABLED`
but deferred this measurement -- GPU was occupied by Phase 2a at the time).

GATE (from docs/research-notes/arc-live-agent-improvement-plan-2026-08-08.md Phase 3a): held-out
change fidelity (HUD-masked, symmetric, per 1a's methodology) on a leave-one-game-out split;
>= 4 of 5 held-out games improve, no live admission-rate regression.

METHOD. Per game: collect a fresh transition pool (`e3.collect_transitions`), split into a
TRAIN slice (shown to the LLM inducer) and a HELD slice (scored only, never shown). Induce twice
-- once with `CARNOT_ARC_INDUCE_PROMPT_ENRICHMENT=0` (off, the shipped default), once with `=1`
(on) -- on the SAME train slice, score both resulting engines' `change_fidelity` on the SAME held
slice via `WorldModelVerifier` with this game's own HUD mask (the same classifier the live
explorer uses, per REQ-ARC-WMTE-6010/6011's four-arm-matrix precedent).

WHY THIS RUNS IN AN ISOLATED ENGINE STORE. `load_engine()` reads from the MUTABLE
`results/arc_e3/<game>/world_model.py` store, which the conductor (running concurrently this
session) also writes to for its own tasks. `project_arc_engine_store_regression` (memory) records
a real incident where an unguarded overwrite destroyed a game's engine. This script MUST be
launched with `CARNOT_ARC_E3_DIR` already set in the environment (read once at import time, not
mutable mid-process -- see `arc_executable_world_model.py`'s own comment on this) to a private,
non-tracked scratch directory, so it never touches the shared store at all.

Sample size: 5 games (the gate's own ">= 4 of 5" language), each scored on 10 held-out
transitions -- enough to compute a real per-game change_fidelity number without the multi-hour
cost a larger held-out set would add; this is a per-game PAIRED comparison (same held set, same
train set, only the flag differs), so the discriminating signal is the WITHIN-GAME delta, not an
absolute-accuracy claim needing large-N power.
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

if not os.environ.get("CARNOT_ARC_E3_DIR"):
    raise SystemExit(
        "set CARNOT_ARC_E3_DIR to a private scratch directory BEFORE launching this script -- "
        "it must never write to the shared results/arc_e3 store while the conductor is running."
    )

from carnot.agentic import arc_executable_world_model as e3  # noqa: E402

OUT = REPO / "results" / "experiment_6246_induce_prompt_enrichment_heldout_ab.json"
CHECKPOINT = Path(
    os.environ.get("CARNOT_EXP6246_CHECKPOINT", "/tmp/carnot_exp6246_checkpoint.json")
)
ROSTER = ("m0r0", "ft09", "tr87", "cn04", "ar25")
N_COLLECT = 50
N_HELD = 10
SEED = 6246
GGUF_REPO_SUBSTR = "gemma-4-31B-it-qat"
CUDA_GPU_INDEX = os.environ.get("CARNOT_ARC_GENERATOR_CUDA_GPU", "1")
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


def _run_arm(prop, game: str, train, held, hud_mask, arm: str) -> dict:
    os.environ["CARNOT_ARC_INDUCE_PROMPT_ENRICHMENT"] = "1" if arm == "on" else "0"
    row: dict = {"game": game, "arm": arm}
    t0 = time.time()
    try:
        ok, detail = prop.induce(game, train, 1)
    except Exception as exc:  # noqa: BLE001
        row.update(induction_ok=False, error=repr(exc)[:200])
        os.environ.pop("CARNOT_ARC_INDUCE_PROMPT_ENRICHMENT", None)
        row["induce_s"] = round(time.time() - t0, 1)
        return row
    row["induce_s"] = round(time.time() - t0, 1)
    row["induction_ok"] = bool(ok)
    if not ok:
        row["induction_failure_detail"] = str(detail)[:200]
        os.environ.pop("CARNOT_ARC_INDUCE_PROMPT_ENRICHMENT", None)
        return row
    try:
        engine, _is_lc = e3.load_engine(game)
        # `hud_mask_enabled=True` EXPLICIT (not the ambient SUBMITTED_WORLD_MODEL_HUD_MASK_ENABLED
        # default): the plan's own Phase 3a gate says "HUD-masked, symmetric, per 1a", and 1a's own
        # REQ-ARC-WMTE-6233 methodology was a MEASUREMENT pass with masking explicitly forced on,
        # not a read of whatever the scored default happens to be today.
        vr = e3.WorldModelVerifier(held, hud_mask=hud_mask, hud_mask_enabled=True).score(engine)
        row["change_fidelity"] = round(float(vr.change_fidelity), 4)
        row["accuracy"] = round(float(vr.accuracy), 4)
        row["cell_recall"] = round(float(vr.cell_recall), 4)
        row["hud_mask_status"] = vr.hud_mask_status
    except Exception as exc:  # noqa: BLE001
        row["score_error"] = repr(exc)[:200]
    os.environ.pop("CARNOT_ARC_INDUCE_PROMPT_ENRICHMENT", None)
    return row


def build_artifact() -> dict:
    t0 = time.time()
    os.environ["CARNOT_ARC_GENERATOR_CUDA_GPU"] = CUDA_GPU_INDEX
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
    done_keys = {(r["game"], r["arm"]) for r in rows}

    for game in ROSTER:
        try:
            trans, cell = e3.collect_transitions(game, n=N_COLLECT, seed=SEED)
        except Exception as exc:  # noqa: BLE001
            rows.append({"game": game, "arm": "collect", "error": repr(exc)[:200]})
            done["rows"] = rows
            _save_checkpoint(done)
            continue
        train, held = trans[:-N_HELD], trans[-N_HELD:]
        hud_mask = _logical_hud_mask(game, cell)
        for arm in ("off", "on"):
            if (game, arm) in done_keys:
                continue
            row = _run_arm(prop, game, train, held, hud_mask, arm)
            rows.append(row)
            done["rows"] = rows
            _save_checkpoint(done)
            print(f"[exp6246] {game} {arm}: {row}", flush=True)

    per_game: dict = {}
    for r in rows:
        if r.get("arm") in ("off", "on"):
            per_game.setdefault(r["game"], {})[r["arm"]] = r

    n_improve = 0
    n_comparable = 0
    game_deltas = []
    for game, arms in per_game.items():
        off, on = arms.get("off"), arms.get("on")
        if not off or not on:
            continue
        off_fid = off.get("change_fidelity")
        on_fid = on.get("change_fidelity")
        if off_fid is None or on_fid is None:
            continue
        n_comparable += 1
        delta = on_fid - off_fid
        game_deltas.append({"game": game, "off": off_fid, "on": on_fid, "delta": round(delta, 4)})
        if delta > 0:
            n_improve += 1

    gate_met = n_comparable > 0 and n_improve >= 4 and n_comparable >= 5

    art = {
        "experiment": "experiment_6246_induce_prompt_enrichment_heldout_ab",
        "title": (
            "Phase 3a gate: leave-one-game-out held-out change-fidelity A/B for "
            "CARNOT_ARC_INDUCE_PROMPT_ENRICHMENT"
        ),
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "roster": list(ROSTER),
        "per_game_results": rows,
        "game_deltas": game_deltas,
        "n_games_comparable": n_comparable,
        "n_games_improved": n_improve,
        "gate_condition": ">=4 of 5 held-out games improve change_fidelity (off->on)",
        "gate_met": gate_met,
        "solve_provenance": "development_proxy",
        "arc_solve_claim": False,
        "verifier_is_oracle": True,
        "verifier_is_oracle_principle": (
            "change_fidelity is scored against the offline game's own real next_grid; this "
            "measures whether the enrichment lever moves the induction-quality metric, not an "
            "oracle-distinct capability claim."
        ),
        "inference_substrate": "live_llm_inference",
        "inference_substrate_principle": (
            "both arms invoke the real gemma-4-31B-it-qat server on a 3090; 60s duration floor "
            "applies to the whole run."
        ),
        "model_specs": [{"name": GGUF_REPO_SUBSTR, "role": "induction proposer, both arms"}],
        "random_seed": SEED,
    }
    art["honest_verdict"] = (
        f"complete_prompt_enrichment_ab_{n_improve}_of_{n_comparable}_games_improved_"
        f"gate_{'met' if gate_met else 'not_met'}"
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
    if OUT.exists() and art.get("gate_met") is not None:
        CHECKPOINT.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

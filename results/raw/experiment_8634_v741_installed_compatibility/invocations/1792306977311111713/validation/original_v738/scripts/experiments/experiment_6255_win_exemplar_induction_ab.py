#!/usr/bin/env python3
"""REQ-ARC-WMTE-6255: does supplying a WIN EXEMPLAR improve induction quality?

THE GAP THIS TESTS. `induce_prompt` has dedicated slots for `win_transition` and
`previous_level_complete_grid`. It is built to show the model what winning looks like.
Every offline induction experiment this project has run passed NEITHER, because
`collect_transitions` produces no level-up to pass -- a salience-biased random walk
essentially never reaches one (measured 0 of 200 steps on dc22 and on cn04). So every
offline induction number we hold, including exp5764's 0.378 pooled held-out, was measured
with that slot empty. Nobody has ever measured what filling it does.

THE DESIGN. One induce call per arm per game, paired, same TRAIN/VALID/HELD split, same
budget, same generator. The ONLY difference is whether `win_transition` and
`previous_level_complete_grid` are passed. Metric: held-out `change_fidelity` of the
induced engine.

WHERE THE EXEMPLAR COMES FROM, AND WHY THAT LIMITS THE CLAIM. `replay_win_transition`
replays a BANKED solve through the game's own `GameAdapter`. That is a DEVELOPMENT PROXY:
a hidden game has no banked solve and no adapter, so the live agent cannot get an exemplar
this way. On the live path the exemplar comes from the agent's own play, so it exists from
level 2 onward and never for level 1. A positive here therefore means "the prompt slot is
worth filling when an exemplar exists" -- it does NOT mean the live agent can obtain one on
a hidden level 1. See the ARC Live-Path Reachability Discipline.

WHAT A NEGATIVE WOULD MEAN, stated before the run. If filling the slot changes nothing,
that is informative too: it would say the induce prompt's win-exemplar section is inert as
written, and the slot is not the lever. Either way this is cheap and the question is
currently unanswered.

ISOLATION. `CARNOT_ARC_E3_DIR` must be a private scratch directory (the exp6247
shared-store clobber incident).
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

from carnot.agentic import arc_executable_world_model as e3  # noqa: E402
from carnot.agentic import arc_rex_refinement as rex  # noqa: E402

OUT = REPO / "results" / "experiment_6255_win_exemplar_induction_ab.json"
CHECKPOINT = Path(
    os.environ.get("CARNOT_EXP6255_CHECKPOINT", "/tmp/carnot_exp6255_checkpoint.json")
)
# Games that have BOTH a banked solve and an adapter, so an exemplar can be replayed.
ROSTER = ("dc22", "cn04", "ls20", "s5i5")
N_COLLECT = 60
N_VALID = 10
N_HELD = 10
SEED = 6255
GGUF_REPO_SUBSTR = "gemma-4-31B-it-qat"
SERVER_PORT = 8943
SHARED_MAX_TOKENS = 16384
INDUCE_TIMEOUT_S = 1500
GATE_MIN_GAMES_IMPROVED = 3


def _frame_hud_mask(game: str):
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_competition_agent import _compute_hud_mask_from_frame

    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    return _compute_hud_mask_from_frame(env.reset())


def _logical_hud_mask(game: str, cell: int):
    frame_mask = _frame_hud_mask(game)
    return None if frame_mask is None else e3.logical_hud_mask(frame_mask, cell)


def _score(source: str, transitions, hud_mask) -> float | None:
    if not source:
        return None
    try:
        engine = rex.load_engine_from_source(source, tag="score")
    except Exception:  # noqa: BLE001
        return 0.0
    vr = e3.WorldModelVerifier(transitions, hud_mask=hud_mask, hud_mask_enabled=True).score(engine)
    return round(float(vr.change_fidelity), 4)


def _induce_arm(prop, game, train, cell, *, win) -> dict:
    """One induce call. `win` is the exemplar transition, or None for the control arm."""
    store = Path(os.environ["CARNOT_ARC_E3_DIR"]) / game / "world_model.py"
    if store.exists():
        store.unlink()  # never let the previous arm's engine be read as this arm's output
    t0 = time.time()
    kwargs = {}
    if win is not None:
        kwargs["win_transition"] = win
        kwargs["previous_level_complete_grid"] = win.next_grid
    try:
        ok, detail = prop.induce(game, list(train), cell, **kwargs)
    except Exception as exc:  # noqa: BLE001
        ok, detail = False, repr(exc)[:200]
    src = store.read_text() if store.exists() else None
    return {
        "ok": bool(ok),
        "detail": str(detail)[:200],
        "source": src,
        "wall_s": round(time.time() - t0, 1),
    }


def build_artifact() -> dict:
    t0 = time.time()
    os.environ.setdefault("CARNOT_ARC_GENERATOR_CUDA_GPU", "0,1")
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
        if not prop._ensure_server():
            return {"honest_verdict": "complete_blocked_cuda_server_failed_to_start"}
    except GeneratorCudaRequiredError as exc:
        return {"honest_verdict": f"complete_blocked_cuda_unavailable_{exc!r}"[:200]}

    done = json.loads(CHECKPOINT.read_text()) if CHECKPOINT.exists() else {}
    rows = list(done.get("rows", []))
    seen = {r["game"] for r in rows}

    for game in ROSTER:
        if game in seen:
            continue
        row: dict = {"game": game}
        trans, cell = e3.collect_transitions(game, n=N_COLLECT, seed=SEED)
        train = trans[: -(N_VALID + N_HELD)]
        held = trans[-N_HELD:]
        hud_mask = _logical_hud_mask(game, cell)

        win = e3.replay_win_transition(game, cell)
        row["win_exemplar_available"] = win is not None
        if win is None:
            # A missing exemplar makes the pair unmeasurable. Record and skip -- never
            # substitute a fabricated one, and never score the control against nothing.
            row["skipped"] = "no banked solve or adapter produced a level-up transition"
            rows.append(row)
            done["rows"] = rows
            CHECKPOINT.write_text(json.dumps(done, indent=2, default=str))
            print(f"[exp6255] {game}: SKIPPED (no win exemplar)", flush=True)
            continue

        control = _induce_arm(prop, game, train, cell, win=None)
        treatment = _induce_arm(prop, game, train, cell, win=win)
        row["control_held"] = _score(control["source"], held, hud_mask)
        row["treatment_held"] = _score(treatment["source"], held, hud_mask)
        row["control_wall_s"] = control["wall_s"]
        row["treatment_wall_s"] = treatment["wall_s"]
        row["control_ok"] = control["ok"]
        row["treatment_ok"] = treatment["ok"]
        row["win_level"] = f"{win.level_before}->{win.level_after}"
        if row["control_held"] is not None and row["treatment_held"] is not None:
            row["delta"] = round(row["treatment_held"] - row["control_held"], 4)
        rows.append(row)
        done["rows"] = rows
        CHECKPOINT.write_text(json.dumps(done, indent=2, default=str))
        print(
            f"[exp6255] {game}: control={row['control_held']} treatment={row['treatment_held']} "
            f"delta={row.get('delta')}",
            flush=True,
        )

    comparable = [r for r in rows if r.get("delta") is not None]
    n = len(comparable)
    n_improved = sum(1 for r in comparable if r["delta"] > 0)
    n_worse = sum(1 for r in comparable if r["delta"] < 0)
    pooled = round(sum(r["delta"] for r in comparable) / n, 4) if n else None
    gate_met = bool(n and n_improved >= GATE_MIN_GAMES_IMPROVED and pooled and pooled > 0)

    art = {
        "experiment": "experiment_6255_win_exemplar_induction_ab",
        "title": "Win-exemplar induction A/B: does filling induce_prompt's win slot help?",
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "roster": list(ROSTER),
        "per_game_results": rows,
        "n_games_comparable": n,
        "n_games_improved": n_improved,
        "n_games_worse": n_worse,
        "pooled_mean_delta": pooled,
        "gate_condition": (
            f">= {GATE_MIN_GAMES_IMPROVED} of the comparable games improve AND pooled delta > 0"
        ),
        "gate_min_games_improved": GATE_MIN_GAMES_IMPROVED,
        "gate_met": gate_met,
        "why_this_was_never_measured": (
            "collect_transitions yields no level-up under a salience-biased random walk "
            "(0 of 200 steps on dc22 and cn04), so every prior offline induction run passed "
            "neither win_transition nor previous_level_complete_grid"
        ),
        "exemplar_source": "replay_win_transition: banked solve replayed through the game's GameAdapter",
        "development_proxy_limit": (
            "a hidden game has no banked solve and no adapter, so the LIVE agent cannot obtain "
            "an exemplar this way. On the live path it comes from the agent's own play and so "
            "exists from level 2 onward, never for level 1. A positive here means the prompt "
            "slot is worth filling WHEN an exemplar exists; it does not show the live agent can "
            "get one on a hidden level 1."
        ),
        "sample_size_note": (
            "at most 4 games, far below the project's n>=30 bar for a percentage-point claim. "
            "Directional signal only."
        ),
        "solve_provenance": "development_proxy",
        "arc_solve_claim": False,
        "verifier_is_oracle": False,
        "verifier_is_oracle_principle": (
            "held-out change_fidelity is an oracle-distinct graded dynamics score, not the "
            "executable win-condition oracle; no level is claimed"
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
            f"complete_win_exemplar_gate_met_{n_improved}_of_{n}_improved_pooled_delta_{pooled}"
        )
    else:
        art["honest_verdict"] = (
            f"complete_win_exemplar_gate_not_met_{n_improved}_improved_{n_worse}_worse_of_{n}_"
            f"pooled_delta_{pooled}"
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

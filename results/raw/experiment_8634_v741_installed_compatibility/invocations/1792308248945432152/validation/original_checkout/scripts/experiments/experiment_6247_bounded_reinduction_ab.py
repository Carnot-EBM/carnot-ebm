#!/usr/bin/env python3
"""REQ-ARC-WMTE-6247: the live-path A/B Phase 4d's own gate requires for
`CARNOT_ARC_BOUNDED_REINDUCTION` (shipped 2026-08-08, commit `e73efa7c85`, never measured).

WHAT THE LEVER IS. `_should_enter_induction` latches `self.induced=True` after the FIRST induction
attempt per level (successful or refused) and never re-enters induction for that level again --
even though a still-stalled agent keeps accumulating exactly the transition evidence a second
attempt would need. `CARNOT_ARC_BOUNDED_REINDUCTION=1` allows a bounded re-attempt
(`reason="renewed_stall_reinduction"`) once >=200 NEW transitions have accumulated since the last
attempt, capped at 3 attempts/level.

WHY THIS NEEDS A FULL LIVE EPISODE, NOT A PER-CELL COMPARISON (unlike REQ-ARC-WMTE-6242/6246).
The lever only has a chance to fire if an episode genuinely stalls after its first induction
attempt, keeps exploring, and accumulates 200+ more transitions without winning -- a property of a
whole multi-hundred-action run, not a single induce call. Reuses
`scripts/arc_scored_path_lever_harness.py`'s `run_cell` (the mature, already-instrumented SCORED-
PATH harness: LLM genuinely on, per-lever fire counters, generator-liveness witnesses) directly,
toggling ONLY `CARNOT_ARC_BOUNDED_REINDUCTION` around each cell -- no new harness invented.

ROSTER + BUDGET. ka59 and re86 (REQ-ARC-WMTE-6244's Mode A games -- both have known induction
difficulties on this stack, making them plausible candidates for a genuine multi-hundred-action
stall the lever could act on). Budget 1500 (per the harness's own module docstring: "a budget
below ~2000 structurally cannot see most of the signal"; 1500 is a deliberate compromise between
giving the lever a real chance to fire and keeping wall-clock bounded for a 2-game x 2-arm run
under live LLM think-mode induction).

THE FIRE-COUNTER DISCIPLINE THIS SCRIPT APPLIES ON TOP OF THE HARNESS'S OWN. A cell where
`induction_reasons.get("renewed_stall_reinduction", 0) == 0` under the flag ON is NOT evidence the
lever doesn't help -- it is evidence the lever never got a chance to fire in THIS cell (the episode
won before 200 new transitions accumulated, or never stalled at all, or hit the 3-attempt cap on
ordinary level-up reinductions first). Every result is classified FIRED vs NOT_FIRED before any
levels/actions comparison is drawn from it, per the project's own "the observe-channel lesson"
already stated in the harness's module docstring.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "python"))

OUT = REPO / "results" / "experiment_6247_bounded_reinduction_ab.json"
CHECKPOINT = Path(
    os.environ.get("CARNOT_EXP6247_CHECKPOINT", "/tmp/carnot_exp6247_checkpoint.json")
)
ROSTER = ("ka59", "re86")
SEED = 20260809
BUDGET = 1500
PORT = 8940


def _load_checkpoint() -> dict:
    if CHECKPOINT.exists():
        return json.loads(CHECKPOINT.read_text())
    return {}


def _save_checkpoint(done: dict) -> None:
    CHECKPOINT.write_text(json.dumps(done, indent=2, default=str))


def _build_current_live_proposer(port: int):
    """The harness's own `build_proposer` hardcodes `repo_substr="Qwen3.5-9B-MTP"` -- the
    RETIRED generator (`project_arc_live_generator` memory: operator 2026-07-28 flipped to
    gemma-4-31B-it-qat, 11-0-2 p=0.00098 over the 9B/27B). Using the harness's helper unmodified
    would measure this lever against a generator the live agent no longer runs. Reuses the
    harness's own `InstrumentedProposer` wrapper (it patches the delegation gap where the agent's
    high-level `induce()` calls resolve to the INNER instance's `generate`, which a naive wrapper
    silently undercounts to zero -- the harness's own docstring records exactly that bug on its
    first smoke run) around a fresh `LocalGGUFProposer` pinned to the CURRENT live generator."""
    from arc_scored_path_lever_harness import InstrumentedProposer
    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

    inner = LocalGGUFProposer(
        repo_substr="gemma-4-31B-it-qat",
        port=port,
        mtp=False,
        kv_quant="q8_0",
        max_tokens=16384,
        no_think_prefix="",
        timeout=1500,
    )
    return InstrumentedProposer(inner)


def build_artifact() -> dict:
    t0 = time.time()
    assert os.environ.get("CARNOT_ARC_DISABLE_INDUCTION") is None, (
        "CARNOT_ARC_DISABLE_INDUCTION must be unset -- this is an LLM-on measurement"
    )
    # FIXED 2026-08-09 (found during the isolation-retry run): this script previously did NOT
    # override CARNOT_ARC_E3_DIR, so `induce()` wrote its induced engines straight into the
    # SHARED `results/arc_e3/<game>/world_model.py` store -- the conductor's own accumulated
    # state -- unconditionally overwriting it on every cell. This is the exact destructive
    # pattern `project_arc_engine_store_regression` warns about (unconditional overwrite
    # destroys retained value). Caught only because the conductor happened to be stopped for
    # this run's isolation test; the two clobbered files were restored via `git checkout --`
    # before commit. Isolating to a private scratch dir makes this script safe to re-run at
    # any time, conductor running or not.
    os.environ.setdefault("CARNOT_ARC_E3_DIR", str(REPO / "results" / "arc_e3_exp6247_scratch"))
    from arc_scored_path_lever_harness import run_cell

    os.environ["CARNOT_ARC_GENERATOR_CUDA_GPU"] = os.environ.get(
        "CARNOT_ARC_GENERATOR_CUDA_GPU", "1"
    )
    os.environ["CARNOT_ARC_GENERATOR_REQUIRE_CUDA"] = "1"
    proposer = _build_current_live_proposer(PORT)
    server_ok = proposer._inner._ensure_server()
    if not server_ok:
        return {"honest_verdict": "complete_blocked_cuda_server_failed_to_start"}

    done = _load_checkpoint()
    rows = list(done.get("rows", []))
    # A cell whose OWN llm_on_row_valid is False (the harness's own reaper-detection witness: the
    # server died mid-episode, so induction never got a real LLM chance) is NOT usable evidence --
    # do not let the checkpoint-resume logic treat it as permanently done. Only a row that either
    # has no `llm` block (llm=False cells, not used here) or explicitly validated is "done".
    valid_done_keys = {
        (r["game"], r["arm"]) for r in rows if r.get("llm_on_row_valid", True) is not False
    }
    rows = [r for r in rows if (r["game"], r["arm"]) in valid_done_keys]

    MAX_CELL_ATTEMPTS = 3
    for game in ROSTER:
        for arm in ("off", "on"):
            if (game, arm) in valid_done_keys:
                continue
            os.environ["CARNOT_ARC_BOUNDED_REINDUCTION"] = "1" if arm == "on" else "0"
            row = None
            for attempt in range(1, MAX_CELL_ATTEMPTS + 1):
                # `_ensure_server()` ALREADY does "if healthy: reuse; else: relaunch" internally
                # (arc_executable_world_model.py:6156) -- calling it unconditionally before every
                # attempt is a safe no-op when the server is fine and a real relaunch when it is
                # not. A SEPARATE outer `_healthy()` pre-check here was tried first and REMOVED: it
                # stacks a second call to the SAME bare-2-second-timeout probe `_ensure_server()`
                # already uses internally (the exact flakiness the harness's own `forbid_spawn`
                # docstring names -- a loaded box can fail a 2s probe on a server that is perfectly
                # alive), and under load that produced two INDEPENDENT servers within the same
                # minute, colliding (a self-inflicted server storm, observed directly: 4 server
                # logs created within a 4-minute window here, several dying mid-model-load with no
                # error at all -- consistent with a second spawn contending for the same port/VRAM
                # while the first was still loading, not an external reaper).
                proposer._inner._ensure_server()
                try:
                    row = run_cell(
                        game,
                        SEED,
                        budget=BUDGET,
                        proposer=proposer,
                        llm=True,
                        arm=f"bounded_reinduction_{arm}",
                    )
                except Exception as exc:  # noqa: BLE001
                    row = {"game": game, "arm": arm, "ran": False, "reason": repr(exc)[:200]}
                else:
                    row["arm"] = arm
                    row["game"] = game
                row["cell_attempts_used"] = attempt
                valid = row.get("llm_on_row_valid", True) is not False
                print(
                    f"[exp6247] {game} {arm} attempt {attempt}/{MAX_CELL_ATTEMPTS}: "
                    f"ran={row.get('ran')} valid={valid} "
                    f"levels={row.get('reached_any_level')} actions_used={row.get('actions')} "
                    f"induction_reasons={row.get('induction_reasons')}",
                    flush=True,
                )
                if valid or attempt == MAX_CELL_ATTEMPTS:
                    break
                print(
                    f"[exp6247] {game} {arm}: invalid LLM row (reaper?), retrying whole cell",
                    flush=True,
                )
            os.environ.pop("CARNOT_ARC_BOUNDED_REINDUCTION", None)
            rows.append(row)
            done["rows"] = rows
            _save_checkpoint(done)

    per_game: dict = {}
    for r in rows:
        per_game.setdefault(r["game"], {})[r["arm"]] = r

    game_summaries = []
    for game, arms in per_game.items():
        off, on = arms.get("off"), arms.get("on")
        fired = bool((on or {}).get("induction_reasons", {}).get("renewed_stall_reinduction"))
        game_summaries.append(
            {
                "game": game,
                "lever_fired_on_arm": fired,
                "off_reached_any_level": (off or {}).get("reached_any_level"),
                "on_reached_any_level": (on or {}).get("reached_any_level"),
                "off_actions": (off or {}).get("actions"),
                "on_actions": (on or {}).get("actions"),
                "off_induction_reasons": (off or {}).get("induction_reasons"),
                "on_induction_reasons": (on or {}).get("induction_reasons"),
            }
        )

    n_fired = sum(1 for g in game_summaries if g["lever_fired_on_arm"])

    art = {
        "experiment": "experiment_6247_bounded_reinduction_ab",
        "title": "Phase 4d gate: live-path A/B for CARNOT_ARC_BOUNDED_REINDUCTION",
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "roster": list(ROSTER),
        "budget": BUDGET,
        "per_game_results": rows,
        "game_summaries": game_summaries,
        "n_games_lever_fired": n_fired,
        "n_games_total": len(game_summaries),
        "solve_provenance": "development_proxy",
        "arc_solve_claim": False,
        "verifier_is_oracle": False,
        "verifier_is_oracle_principle": (
            "banked levels on the offline arcade are the live agent's own real outcome, not a "
            "circular self-check."
        ),
        "inference_substrate": "live_llm_inference",
        "random_seed": SEED,
    }
    if n_fired == 0:
        art["honest_verdict"] = (
            "complete_lever_never_fired_in_either_game_uninterpretable_not_a_negative_result"
        )
    else:
        n_improved = sum(
            1
            for g in game_summaries
            if g["lever_fired_on_arm"]
            and (g["on_reached_any_level"] or False)
            and not (g["off_reached_any_level"] or False)
        )
        art["n_games_improved_where_fired"] = n_improved
        art["honest_verdict"] = (
            f"complete_bounded_reinduction_ab_fired_{n_fired}_of_{len(game_summaries)}_games_"
            f"improved_{n_improved}_where_fired"
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

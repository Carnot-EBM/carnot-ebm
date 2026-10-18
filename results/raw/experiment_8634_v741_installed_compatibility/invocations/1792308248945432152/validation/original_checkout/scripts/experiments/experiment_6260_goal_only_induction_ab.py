#!/usr/bin/env python3
"""REQ-ARC-WMTE-6260: does a FOCUSED, evidence-carrying goal prompt beat the combined call?

THE ONE UNTESTED LAYER. Every lever tried so far attacked the wrong layer. exp6255/6256 added
a win exemplar to the prompt's framing (no effect on the predicate). exp6258/6259 changed
filtering (nothing admissible to filter). Neither changed how the predicate is PRODUCED. Today
`induce()` asks for `engine` AND `is_level_complete` in ONE code block, so the model splits its
budget between learning dynamics and defining the win -- and the win half came out degenerate
22 times out of 22 across exp6256 and exp6259.

`_goal_only_prompt` already exists and is documented as "a FOCUSED is_level_complete-only
prompt, so the model spends its whole budget on the win condition (not the engine)". It fires
only as a fallback today and nobody has scored what it produces.

WHY THE TRANSITIONS FLAG IS ON HERE. `_goal_prompt_transitions_on`'s own docstring records that
the goal-only prompt is "the evidence-free prompt in the pair", and that the 2026-08-01 taxonomy
traced 12 of 13 whole-board "every cell is one colour" predicates to exactly the cells it
produced. Running the focused prompt WITHOUT evidence would re-measure a known failure. The
transitions are the agent's own observations, so showing them crosses no line.

THE HONEST NARROWING, taken from that same docstring. It records that the 2026-08-02 A/B scored
this flag against windows cut from a BANKED WINNING ROUTE, and that whether it helps on
exploration-buffer transitions is OPEN. This experiment inherits that limit: it uses transitions
from `collect_transitions`, a salience-biased random walk, which is closer to an exploration
buffer than to a winning route -- and which never reaches a level-up. So this measures the
focused prompt on the evidence a hidden-game agent would plausibly have, not on a solved
route's.

SCORED TWO-SIDED, because one side cannot fail. Specificity over held-out transitions is a
false-positive check that a constant-False predicate passes perfectly (REQ-ARC-WMTE-6257).
Sensitivity -- does it fire on a grid where a level-up REALLY happened -- is the side that
catches the degenerate case. A single non-degenerate predicate here would be the first crack in
a wall that has held against four other levers.

ISOLATION. `CARNOT_ARC_E3_DIR` must be a private scratch directory (the exp6247 incident).
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
# The focused prompt is evidence-free by default; running it that way would re-measure a
# documented failure rather than test the mechanism. See the module docstring.
os.environ["CARNOT_ARC_GOAL_PROMPT_TRANSITIONS"] = "1"

import numpy as np  # noqa: E402

from carnot.agentic import arc_executable_world_model as e3  # noqa: E402

OUT = REPO / "results" / "experiment_6260_goal_only_induction_ab.json"
CHECKPOINT = Path(
    os.environ.get("CARNOT_EXP6260_CHECKPOINT", "/tmp/carnot_exp6260_checkpoint.json")
)
ROSTER = ("dc22", "cn04", "ls20", "s5i5")
N_COLLECT = 60
N_HELD = 10
SEED = 6260
GGUF_REPO_SUBSTR = "gemma-4-31B-it-qat"
SERVER_PORT = 8946
SHARED_MAX_TOKENS = 16384
INDUCE_TIMEOUT_S = 1500


def _frame_hud_mask(game: str):
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_competition_agent import _compute_hud_mask_from_frame

    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    return _compute_hud_mask_from_frame(env.reset())


def _logical_hud_mask(game: str, cell: int):
    m = _frame_hud_mask(game)
    return None if m is None else e3.logical_hud_mask(m, cell)


def _load_goal(source: str, tag: str):
    import importlib.util
    import tempfile

    with tempfile.NamedTemporaryFile(
        "w", suffix=".py", prefix=f"exp6260_{tag}_", delete=False
    ) as f:
        f.write(source)
        path = Path(f.name)
    try:
        spec = importlib.util.spec_from_file_location(f"exp6260_{tag}_{path.stem}", path)
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return getattr(mod, "engine", None), getattr(mod, "is_level_complete", None)
    finally:
        path.unlink(missing_ok=True)


def _score_goal(source: str | None, tag: str, held, win) -> dict:
    out = {
        "predicate_present": False,
        "specificity": None,
        "fires_on_real_win": None,
        "is_degenerate_constant_false": None,
        "fires_on_start_grid": None,
    }
    if not source:
        return out
    try:
        engine, goal = _load_goal(source, tag)
    except Exception as exc:  # noqa: BLE001
        out["load_error"] = repr(exc)[:160]
        return out
    if goal is None:
        return out
    out["predicate_present"] = True
    try:
        gc = e3.score_goal_predicate_consistency(
            goal, held, engine=engine, win_grids=[win.next_grid]
        )
        out["specificity"] = round(float(gc.accuracy), 4)
        out["fires_on_real_win"] = bool(gc.sensitivity_win_grids_fired > 0)
        out["is_degenerate_constant_false"] = gc.is_degenerate_constant_false
    except Exception as exc:  # noqa: BLE001
        out["score_error"] = repr(exc)[:160]
    # Separates "constant False" from "constant True": a predicate firing on the opening grid
    # as well as the win is degenerate in the other direction.
    try:
        out["fires_on_start_grid"] = bool(goal(np.asarray(held[0].grid)))
    except Exception:  # noqa: BLE001
        pass
    return out


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
    store = None

    for game in ROSTER:
        if game in seen:
            continue
        row: dict = {"game": game}
        trans, cell = e3.collect_transitions(game, n=N_COLLECT, seed=SEED)
        train = trans[: -(2 * N_HELD)]
        held = trans[-N_HELD:]
        _ = _logical_hud_mask(game, cell)
        win = e3.replay_win_transition(game, cell)
        if win is None:
            row["skipped"] = (
                "no win grid -- sensitivity is unmeasurable, so neither arm is scorable"
            )
            rows.append(row)
            done["rows"] = rows
            CHECKPOINT.write_text(json.dumps(done, indent=2, default=str))
            print(f"[exp6260] {game}: SKIPPED (no win grid)", flush=True)
            continue

        # ARM A: the current default -- one combined engine + is_level_complete call.
        store = Path(os.environ["CARNOT_ARC_E3_DIR"]) / game / "world_model.py"
        if store.exists():
            store.unlink()
        ta = time.time()
        try:
            prop.induce(game, list(train), cell)
        except Exception as exc:  # noqa: BLE001
            row["arm_a_error"] = repr(exc)[:160]
        src_a = store.read_text() if store.exists() else None
        row["arm_a_wall_s"] = round(time.time() - ta, 1)

        # ARM B: the FOCUSED goal-only prompt, carrying the agent's own transitions.
        tb = time.time()
        src_b = None
        try:
            # `previous_level_complete_grid=None` ON PURPOSE. Passing `win.next_grid` here
            # would put the very grid sensitivity is scored against INTO the prompt, so the
            # model could write `grid == <that grid>` and "fire on the real win" by
            # memorisation. That would measure leakage, not induction. None is also the
            # realistic hidden-game level-1 state: the agent has never won, so no previous
            # level-complete grid exists.
            goal_prompt = prop._goal_only_prompt(game, None, list(train))
            ok_b, code_b = prop.generate(
                goal_prompt, ("is_level_complete",), tries=prop.tries, codeonly_eligible=True
            )
            if ok_b:
                src_b = code_b
        except Exception as exc:  # noqa: BLE001
            row["arm_b_error"] = repr(exc)[:160]
        row["arm_b_wall_s"] = round(time.time() - tb, 1)

        row["arm_a"] = _score_goal(src_a, f"{game}_a", held, win)
        row["arm_b"] = _score_goal(src_b, f"{game}_b", held, win)
        row["arm_a_source"] = src_a
        row["arm_b_source"] = src_b
        rows.append(row)
        done["rows"] = rows
        CHECKPOINT.write_text(json.dumps(done, indent=2, default=str))
        a, b = row["arm_a"], row["arm_b"]
        print(
            f"[exp6260] {game}: combined fires={a['fires_on_real_win']} spec={a['specificity']} "
            f"| goal_only fires={b['fires_on_real_win']} spec={b['specificity']}",
            flush=True,
        )

    comparable = [r for r in rows if r.get("arm_a") and r.get("arm_b")]
    n = len(comparable)
    a_fires = sum(1 for r in comparable if r["arm_a"].get("fires_on_real_win"))
    b_fires = sum(1 for r in comparable if r["arm_b"].get("fires_on_real_win"))
    b_present = sum(1 for r in comparable if r["arm_b"].get("predicate_present"))
    gained = [
        r["game"]
        for r in comparable
        if r["arm_b"].get("fires_on_real_win") and not r["arm_a"].get("fires_on_real_win")
    ]

    art = {
        "experiment": "experiment_6260_goal_only_induction_ab",
        "title": "Focused evidence-carrying goal-only induction vs the combined call",
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "roster": list(ROSTER),
        "per_game_results": rows,
        "n_games_comparable": n,
        "combined_call_predicates_firing_on_real_win": a_fires,
        "goal_only_predicates_firing_on_real_win": b_fires,
        "goal_only_predicates_produced": b_present,
        "games_where_goal_only_gained_sensitivity": gained,
        "gate_condition": (
            "the focused arm produces at least ONE predicate that fires on a real win where the "
            "combined arm's did not. 22 of 22 prior inductions produced none, so a single "
            "non-degenerate predicate is the result worth having."
        ),
        "gate_met": bool(gained),
        "why_transitions_are_on": (
            "_goal_prompt_transitions_on's docstring records the focused prompt as the "
            "evidence-free one, and traces 12 of 13 whole-board one-colour predicates to the "
            "cells it produced. Running it evidence-free would re-measure a known failure."
        ),
        "inherited_open_question": (
            "that same docstring narrows the 2026-08-02 A/B of this flag: it was scored against "
            "windows cut from a BANKED WINNING ROUTE, and whether the flag helps on "
            "exploration-buffer transitions is OPEN. This run uses collect_transitions output, a "
            "salience-biased random walk containing no level-up, so it measures the weaker and "
            "more realistic distribution."
        ),
        "leakage_control": (
            "the focused prompt is given previous_level_complete_grid=None, NOT the win grid "
            "sensitivity is scored against. Passing that grid would let the model fire on the "
            "real win by memorising it, measuring leakage rather than induction. None is also "
            "the realistic hidden-game level-1 state."
        ),
        "development_proxy_limit": (
            "the win grid used for sensitivity comes from replaying a banked solve through a "
            "GameAdapter. A hidden game has none, so sensitivity is measurable here and NOT on a "
            "hidden level 1. The mechanism under test -- a focused goal prompt -- does transfer; "
            "the ability to SCORE it does not."
        ),
        "sample_size_note": "at most 4 games. Directional only, far below the n>=30 bar.",
        "solve_provenance": "development_proxy",
        "arc_solve_claim": False,
        "verifier_is_oracle": False,
        "verifier_is_oracle_principle": (
            "goal-predicate sign agreement against recorded level-up ground truth is a check on "
            "an induced hypothesis, not the executable win oracle driving a solve; no level is "
            "claimed"
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
    elif gained:
        art["honest_verdict"] = (
            f"complete_goal_only_induction_gate_met_fires_{a_fires}_to_{b_fires}_of_{n}_gained_on_"
            + "_".join(gained)
        )
    else:
        art["honest_verdict"] = (
            f"complete_goal_only_induction_gate_not_met_fires_{a_fires}_to_{b_fires}_of_{n}_"
            f"goal_only_produced_{b_present}"
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

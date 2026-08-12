#!/usr/bin/env python3
"""REQ-ARC-WMTE-6256: does the win exemplar help the GOAL predicate, which is what it is for?

WHY THIS RUN EXISTS. exp6255 measured what happens when `induce_prompt`'s win-exemplar slots
are filled, and found dynamics fidelity got WORSE (1 improved, 3 worse, pooled -0.2476). But
it scored the ENGINE only. The exemplar's purpose is arguably the GOAL predicate
(`is_level_complete`), which exp6255 never scored at all. So the negative may be measuring
the wrong half, and the honest next step is to score the half the exemplar is aimed at.

WHAT IS SCORED, AND WHY TWO SIDES ARE NEEDED. A goal predicate can fail in two opposite ways
and one number hides one of them:

  * SPECIFICITY -- does it stay False on ordinary non-winning states? Measured with
    `score_goal_predicate_consistency` over held-out transitions, whose ground truth is
    `level_after > level_before`. Held-out data contains no level-ups (a random walk never
    reaches one), so this side is purely a false-positive check.
  * SENSITIVITY -- does it fire True on a REAL win grid? Measured directly against the
    replayed level-up transition's `next_grid`.

A predicate that returns False everywhere scores perfectly on specificity and is useless. A
predicate that returns True everywhere fires on the win and is equally useless. Reporting
both is the only way either number means anything. exp6252's retraction earlier in this
batch came from exactly this class of mistake -- a control that could not fail.

The engine is scored too, so this run reproduces exp6255's dynamics comparison on the same
footing rather than asking the reader to compare across runs.

ENGINE SOURCES ARE SAVED THIS TIME. exp6255 discarded them, so answering this question
required re-inducing from scratch. The sources go in the artifact so the next question about
these engines costs no GPU.

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

import numpy as np  # noqa: E402

from carnot.agentic import arc_executable_world_model as e3  # noqa: E402
from carnot.agentic import arc_rex_refinement as rex  # noqa: E402

OUT = REPO / "results" / "experiment_6256_win_exemplar_goal_predicate_ab.json"
CHECKPOINT = Path(
    os.environ.get("CARNOT_EXP6256_CHECKPOINT", "/tmp/carnot_exp6256_checkpoint.json")
)
ROSTER = ("dc22", "cn04", "ls20", "s5i5")
N_COLLECT = 60
N_VALID = 10
N_HELD = 10
SEED = 6256
GGUF_REPO_SUBSTR = "gemma-4-31B-it-qat"
SERVER_PORT = 8944
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


def _load_both(source: str):
    """Import `engine` AND `is_level_complete` from an induced source."""
    import importlib.util
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".py", prefix="exp6256_", delete=False) as f:
        f.write(source)
        path = Path(f.name)
    try:
        spec = importlib.util.spec_from_file_location(f"exp6256_{path.stem}", path)
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return getattr(mod, "engine", None), getattr(mod, "is_level_complete", None)
    finally:
        path.unlink(missing_ok=True)


def _score_arm(source: str | None, held, hud_mask, win) -> dict:
    """Engine dynamics fidelity plus BOTH sides of the goal predicate."""
    out: dict = {
        "engine_held_fidelity": None,
        "goal_specificity_accuracy": None,
        "goal_false_positives": None,
        "goal_fires_on_real_win": None,
        "goal_predicate_present": False,
        "load_error": None,
    }
    if not source:
        return out
    try:
        engine, goal = _load_both(source)
    except Exception as exc:  # noqa: BLE001
        out["load_error"] = repr(exc)[:160]
        return out

    if engine is not None:
        try:
            vr = e3.WorldModelVerifier(held, hud_mask=hud_mask, hud_mask_enabled=True).score(engine)
            out["engine_held_fidelity"] = round(float(vr.change_fidelity), 4)
        except Exception as exc:  # noqa: BLE001
            out["load_error"] = repr(exc)[:160]

    if goal is None:
        return out
    out["goal_predicate_present"] = True

    # SPECIFICITY: held-out transitions contain no level-ups, so every True is a false
    # positive. Scored through the project's own goal-consistency checker rather than a
    # local reimplementation.
    try:
        gc = e3.score_goal_predicate_consistency(goal, held, engine=engine)
        out["goal_specificity_accuracy"] = round(float(gc.accuracy), 4)
        out["goal_false_positives"] = int(gc.n - gc.n_correct)
        out["goal_n_scored"] = int(gc.n)
        out["goal_n_real_levelups_in_held"] = int(gc.n_real_levelups)
    except Exception as exc:  # noqa: BLE001
        out["goal_specificity_error"] = repr(exc)[:160]

    # SENSITIVITY: does it fire on a grid where a level-up REALLY happened?
    try:
        out["goal_fires_on_real_win"] = bool(goal(np.asarray(win.next_grid)))
    except Exception as exc:  # noqa: BLE001
        out["goal_sensitivity_error"] = repr(exc)[:160]
    return out


def _induce(prop, game, train, cell, *, win) -> dict:
    store = Path(os.environ["CARNOT_ARC_E3_DIR"]) / game / "world_model.py"
    if store.exists():
        store.unlink()  # never read the previous arm's engine as this arm's output
    kwargs = {}
    if win is not None:
        kwargs["win_transition"] = win
        kwargs["previous_level_complete_grid"] = win.next_grid
    t0 = time.time()
    try:
        ok, detail = prop.induce(game, list(train), cell, **kwargs)
    except Exception as exc:  # noqa: BLE001
        ok, detail = False, repr(exc)[:200]
    return {
        "ok": bool(ok),
        "detail": str(detail)[:200],
        "source": store.read_text() if store.exists() else None,
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
            row["skipped"] = "no banked solve or adapter produced a level-up transition"
            rows.append(row)
            done["rows"] = rows
            CHECKPOINT.write_text(json.dumps(done, indent=2, default=str))
            print(f"[exp6256] {game}: SKIPPED (no win exemplar)", flush=True)
            continue

        control = _induce(prop, game, train, cell, win=None)
        treatment = _induce(prop, game, train, cell, win=win)
        row["control"] = {
            **_score_arm(control["source"], held, hud_mask, win),
            "ok": control["ok"],
            "wall_s": control["wall_s"],
        }
        row["treatment"] = {
            **_score_arm(treatment["source"], held, hud_mask, win),
            "ok": treatment["ok"],
            "wall_s": treatment["wall_s"],
        }
        # Saved so the next question about these engines costs no GPU -- exp6255 discarded
        # its sources and that is why this run had to re-induce from scratch.
        row["control_source"] = control["source"]
        row["treatment_source"] = treatment["source"]
        rows.append(row)
        done["rows"] = rows
        CHECKPOINT.write_text(json.dumps(done, indent=2, default=str))
        c, t = row["control"], row["treatment"]
        print(
            f"[exp6256] {game}: engine {c['engine_held_fidelity']}->{t['engine_held_fidelity']} | "
            f"goal_spec {c['goal_specificity_accuracy']}->{t['goal_specificity_accuracy']} | "
            f"fires_on_win {c['goal_fires_on_real_win']}->{t['goal_fires_on_real_win']}",
            flush=True,
        )

    comparable = [r for r in rows if r.get("control") and r.get("treatment")]
    n = len(comparable)

    def _sens_gain(r) -> int:
        c = r["control"]["goal_fires_on_real_win"]
        t = r["treatment"]["goal_fires_on_real_win"]
        if c is None or t is None:
            return 0
        return int(bool(t)) - int(bool(c))

    n_sens_gained = sum(1 for r in comparable if _sens_gain(r) > 0)
    n_sens_lost = sum(1 for r in comparable if _sens_gain(r) < 0)
    ctrl_fires = sum(1 for r in comparable if r["control"]["goal_fires_on_real_win"])
    treat_fires = sum(1 for r in comparable if r["treatment"]["goal_fires_on_real_win"])

    def _spec_delta(r):
        c = r["control"]["goal_specificity_accuracy"]
        t = r["treatment"]["goal_specificity_accuracy"]
        return None if c is None or t is None else round(t - c, 4)

    spec_deltas = [d for d in (_spec_delta(r) for r in comparable) if d is not None]
    pooled_spec = round(sum(spec_deltas) / len(spec_deltas), 4) if spec_deltas else None
    gate_met = bool(n and n_sens_gained >= GATE_MIN_GAMES_IMPROVED and n_sens_gained > n_sens_lost)

    art = {
        "experiment": "experiment_6256_win_exemplar_goal_predicate_ab",
        "title": "Win-exemplar A/B scored on the GOAL predicate, both sides",
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "roster": list(ROSTER),
        "per_game_results": rows,
        "n_games_comparable": n,
        "n_games_sensitivity_gained": n_sens_gained,
        "n_games_sensitivity_lost": n_sens_lost,
        "control_predicates_firing_on_real_win": ctrl_fires,
        "treatment_predicates_firing_on_real_win": treat_fires,
        "pooled_specificity_delta": pooled_spec,
        "gate_condition": (
            f">= {GATE_MIN_GAMES_IMPROVED} games gain SENSITIVITY (predicate fires on a real "
            "win where the control's did not) AND gains exceed losses"
        ),
        "gate_met": gate_met,
        "why_two_sided": (
            "held-out transitions contain no level-ups, so specificity alone is a "
            "false-positive check that a return-False-everywhere predicate passes perfectly. "
            "Sensitivity against a real win grid is what makes either number meaningful."
        ),
        "answers_which_question": (
            "exp6255 found the win exemplar HURTS dynamics fidelity but scored the engine only. "
            "This scores the goal predicate, which is what the exemplar is aimed at."
        ),
        "development_proxy_limit": (
            "the exemplar and the win grid both come from replaying a banked solve through the "
            "game's GameAdapter. A hidden game has neither, so nothing here shows the live agent "
            "can obtain an exemplar on a hidden level 1; on the live path one exists only from "
            "level 2 onward, from the agent's own play."
        ),
        "sample_size_note": "at most 4 games, far below the n>=30 bar. Directional only.",
        "solve_provenance": "development_proxy",
        "arc_solve_claim": False,
        "verifier_is_oracle": False,
        "verifier_is_oracle_principle": (
            "goal-predicate sign agreement against observed level_after > level_before is a "
            "recorded-ground-truth check on an induced hypothesis, not the executable "
            "win-condition oracle driving a solve; no level is claimed"
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
            f"complete_win_exemplar_goal_gate_met_sensitivity_gained_{n_sens_gained}_of_{n}_"
            f"fires_{ctrl_fires}_to_{treat_fires}"
        )
    else:
        art["honest_verdict"] = (
            f"complete_win_exemplar_goal_gate_not_met_sensitivity_gained_{n_sens_gained}_lost_"
            f"{n_sens_lost}_of_{n}_fires_{ctrl_fires}_to_{treat_fires}_spec_delta_{pooled_spec}"
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

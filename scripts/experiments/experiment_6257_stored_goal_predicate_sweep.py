#!/usr/bin/env python3
"""REQ-ARC-WMTE-6257: how many STORED goal predicates are degenerate?

WHY. A smoke test found dc22's stored `is_level_complete` scores a perfect 1.0 on
`score_goal_predicate_consistency` while never firing on the actual win state -- the
return-False-everywhere predicate. That happens because `collect_transitions` yields no
level-ups, so held-out data contains only non-winning states, and against that data a
constant-False predicate is 100% correct. The measure is structurally one-sided for every
game, not just dc22. This sweep asks how many of the 25 stored engines have the same
problem.

TWO SIDES, ALWAYS. Specificity is the false-positive rate on held-out non-win states.
Sensitivity is whether the predicate fires on a grid where a level-up REALLY happened,
recovered by `replay_win_transition`. A predicate is DEGENERATE here when it has high
specificity and does not fire on the real win: it looks perfect and decides nothing.

NO LLM, NO GPU. This reads engines already induced and stored, so it runs beside a GPU
experiment. `results/arc_e3` is EVIDENCE -- read only, never written.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
os.environ.setdefault("CARNOT_ARC_E3_DIR", tempfile.mkdtemp(prefix="carnot_exp6257_scratch_"))

import numpy as np  # noqa: E402

from carnot.agentic import arc_executable_world_model as e3  # noqa: E402

OUT = REPO / "results" / "experiment_6257_stored_goal_predicate_sweep.json"
TRACKED_STORE = REPO / "results" / "arc_e3"
NON_GAME_DIRS = {"g", "positive_control_4557"}
N_COLLECT = 40
N_HELD = 10
SEED = 6257


def _load_both(source: str, tag: str):
    with tempfile.NamedTemporaryFile(
        "w", suffix=".py", prefix=f"exp6257_{tag}_", delete=False
    ) as f:
        f.write(source)
        path = Path(f.name)
    try:
        spec = importlib.util.spec_from_file_location(f"exp6257_{tag}_{path.stem}", path)
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return getattr(mod, "engine", None), getattr(mod, "is_level_complete", None)
    finally:
        path.unlink(missing_ok=True)


def _sweep_game(game: str) -> dict:
    row: dict = {"game": game}
    try:
        engine, goal = _load_both((TRACKED_STORE / game / "world_model.py").read_text(), game)
    except Exception as exc:  # noqa: BLE001
        row["error"] = repr(exc)[:160]
        return row
    row["goal_predicate_present"] = goal is not None
    if goal is None:
        return row
    try:
        trans, cell = e3.collect_transitions(game, n=N_COLLECT, seed=SEED)
    except Exception as exc:  # noqa: BLE001
        row["error"] = f"collect_transitions: {exc!r}"[:160]
        return row
    held = trans[-N_HELD:]

    try:
        gc = e3.score_goal_predicate_consistency(goal, held, engine=engine)
        row["specificity_accuracy"] = round(float(gc.accuracy), 4)
        row["false_positives"] = int(gc.n - gc.n_correct)
        row["n_real_levelups_in_held"] = int(gc.n_real_levelups)
    except Exception as exc:  # noqa: BLE001
        row["specificity_error"] = repr(exc)[:160]

    win = e3.replay_win_transition(game, cell)
    row["win_exemplar_available"] = win is not None
    if win is None:
        # Sensitivity is UNMEASURABLE without a real win grid. Record that, never assume.
        row["fires_on_real_win"] = None
        return row
    try:
        row["fires_on_real_win"] = bool(goal(np.asarray(win.next_grid)))
    except Exception as exc:  # noqa: BLE001
        row["sensitivity_error"] = repr(exc)[:160]
        row["fires_on_real_win"] = None

    # Also probe a NON-win grid the predicate has not been scored against, to separate
    # "constant False" from "fires only on the right thing".
    try:
        row["fires_on_start_grid"] = bool(goal(np.asarray(trans[0].grid)))
    except Exception:  # noqa: BLE001
        row["fires_on_start_grid"] = None

    spec = row.get("specificity_accuracy")
    if row["fires_on_real_win"] is False and spec is not None and spec >= 0.9:
        row["classification"] = "DEGENERATE_looks_perfect_never_fires"
    elif row["fires_on_real_win"] and row.get("fires_on_start_grid"):
        row["classification"] = "DEGENERATE_fires_on_everything"
    elif row["fires_on_real_win"]:
        row["classification"] = "DISCRIMINATING_fires_on_win_only"
    else:
        row["classification"] = "unclear"
    return row


def build_artifact() -> dict:
    t0 = time.time()
    roster = sorted(
        d.name
        for d in TRACKED_STORE.iterdir()
        if d.is_dir() and d.name not in NON_GAME_DIRS and (d / "world_model.py").exists()
    )
    rows = []
    for game in roster:
        try:
            row = _sweep_game(game)
        except Exception as exc:  # noqa: BLE001
            row = {"game": game, "error": repr(exc)[:160]}
        rows.append(row)
        print(
            f"[exp6257] {game}: spec={row.get('specificity_accuracy')} "
            f"fires_on_win={row.get('fires_on_real_win')} -> {row.get('classification')}",
            flush=True,
        )

    measurable = [r for r in rows if r.get("fires_on_real_win") is not None]
    degenerate = [
        r for r in measurable if str(r.get("classification", "")).startswith("DEGENERATE")
    ]
    discriminating = [
        r for r in measurable if r.get("classification") == "DISCRIMINATING_fires_on_win_only"
    ]
    perfect_but_dead = [
        r for r in measurable if r.get("classification") == "DEGENERATE_looks_perfect_never_fires"
    ]

    art = {
        "experiment": "experiment_6257_stored_goal_predicate_sweep",
        "title": "How many stored goal predicates look perfect while deciding nothing?",
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "roster": roster,
        "per_game_results": rows,
        "n_games": len(roster),
        "n_measurable": len(measurable),
        "n_degenerate": len(degenerate),
        "n_perfect_specificity_but_never_fires": len(perfect_but_dead),
        "n_discriminating": len(discriminating),
        "degenerate_games": [r["game"] for r in degenerate],
        "discriminating_games": [r["game"] for r in discriminating],
        "why_specificity_alone_is_not_enough": (
            "collect_transitions yields no level-ups, so held-out data holds only non-winning "
            "states and a constant-False predicate scores 100%. Sensitivity against a real win "
            "grid is the only side that can fail such a predicate."
        ),
        "unmeasurable_note": (
            "a game with no banked solve or adapter yields no win grid, so its sensitivity is "
            "UNMEASURABLE and it is excluded from the counts rather than assumed good"
        ),
        "solve_provenance": "development_proxy",
        "arc_solve_claim": False,
        "verifier_is_oracle": False,
        "verifier_is_oracle_principle": (
            "this audits an induced hypothesis against recorded level-up ground truth; it is "
            "not the executable win oracle driving a solve, and no level is claimed"
        ),
        "inference_substrate": "offline_arcade_live_agent_runtime_self_discovery_no_llm",
        "random_seed": SEED,
        "preconditions_checked": [
            {"resource": "tracked_e3_store_readable", "available": TRACKED_STORE.exists()},
            {"resource": "no_llm_required", "available": True},
            {"resource": "no_gpu_required", "available": True},
        ],
    }
    if not measurable:
        art["honest_verdict"] = (
            "complete_blocked_no_game_yielded_a_win_grid_sensitivity_unmeasurable"
        )
    else:
        art["honest_verdict"] = (
            f"complete_stored_goal_predicate_sweep_{len(degenerate)}_of_{len(measurable)}_degenerate_"
            f"{len(perfect_but_dead)}_look_perfect_but_never_fire"
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

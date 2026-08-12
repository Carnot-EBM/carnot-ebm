#!/usr/bin/env python3
"""REQ-ARC-WMTE-6252: does ANY goal gradient beat the flat production default at planning?

WHAT THIS MEASURES. The 2026-08-09 note says a zero-gradient goal signal is a second,
independent gate on the live agent, separate from dynamics-induction quality. The claim
is testable and has never been tested directly. `plan_in_model` accepts an optional
`goal_energy(grid) -> float` (lower = closer) and uses it as the best-first heap key.
The production default supplies no gradient, so the search is plain BFS. This experiment
holds the engine, the goal predicate, and the start grid FIXED, and varies ONLY the heap
key. Any difference is therefore attributable to the gradient alone.

WHY THIS RUNS BEFORE BUILDING A NEW INDUCER. A prior-art sweep found four graded goal
inducers already exist in `arc_agi3_goal_induction.py`, and one of them was already
refuted empirically. Building a fifth before knowing whether the SEARCH can use a
gradient at all would risk a fifth refutation for the same unmeasured reason. If the
flat, graded, and random-ablation arms all tie, then heap ordering is not the lever and
the whole P3 direction should stop. That is a cheap answer to a expensive question.

THE ABLATION IS LOAD-BEARING, NOT DECORATION. `uniform_random` is a deterministic hash
of the grid: it carries ZERO information about the goal, but it still changes the search
ORDER. If a graded arm only matches the random arm, the apparent gain is reordering
noise, not goal signal. The goal-induction doctrine in this project mandates this
control. Read the gate below: a graded arm must beat BOTH flat and random to count.

NO LLM. NO GPU. This reads engines that were already induced and stored, so it can run
beside a GPU experiment without competing for the card.

`results/arc_e3` IS EVIDENCE. This script only ever READS it. `CARNOT_ARC_E3_DIR` is
still pointed at a scratch directory before import, as a second guard, so that no code
path reached from here can write the tracked store.
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

# Point the writable store at scratch BEFORE importing the module that reads the env var
# at import time. The tracked store is read below by explicit absolute path instead.
os.environ.setdefault("CARNOT_ARC_E3_DIR", tempfile.mkdtemp(prefix="carnot_exp6252_scratch_"))

import numpy as np  # noqa: E402

from carnot.agentic import arc_executable_world_model as e3  # noqa: E402
from carnot.agentic.arc_goal_energy_live import make_uniform_goal_energy  # noqa: E402

OUT = REPO / "results" / "experiment_6252_goal_gradient_ab.json"
CHECKPOINT = Path(
    os.environ.get("CARNOT_EXP6252_CHECKPOINT", "/tmp/carnot_exp6252_checkpoint.json")
)
TRACKED_STORE = REPO / "results" / "arc_e3"
# Excluded from the roster: not real games. "g" is a 4-line stub and
# "positive_control_4557" is a synthetic fixture.
NON_GAME_DIRS = {"g", "positive_control_4557"}
N_COLLECT = 40
# Matches `plan_in_model`'s own production default. A smaller cap was tried first and
# every arm hit it without finding a plan on dc22, which measures the cap, not the
# gradient.
MAX_NODES = 20000
SEED = 6252
# Pre-registered gate thresholds.
GATE_MIN_GAMES_BEATEN = 3


def _load_engine_and_goal(source: str, tag: str):
    """Import BOTH `engine` and `is_level_complete` from stored engine source.

    `arc_rex_refinement.load_engine_from_source` returns only the engine. This
    experiment needs the goal predicate too, because the predicate is what TERMINATES
    the search in every arm -- only the heap key varies.
    """
    with tempfile.NamedTemporaryFile(
        "w", suffix=".py", prefix=f"exp6252_{tag}_", delete=False
    ) as f:
        f.write(source)
        path = Path(f.name)
    try:
        spec = importlib.util.spec_from_file_location(f"exp6252_{tag}_{path.stem}", path)
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return getattr(mod, "engine", None), getattr(mod, "is_level_complete", None)
    finally:
        path.unlink(missing_ok=True)


def _novelty_energy(observed_grids):
    """Graded energy: distance to the NEAREST already-observed grid, normalised.

    This is the go-explore intuition, and it is goal-BLIND on purpose. It rewards
    reaching unfamiliar states. It is included because an agent that cannot see the goal
    can still make progress by covering new ground, so it is the honest floor a
    goal-AWARE gradient has to beat.
    """
    stack = [np.asarray(g) for g in observed_grids]

    def energy(grid) -> float:
        g = np.asarray(grid)
        best = 1.0
        for seen in stack:
            if seen.shape != g.shape:
                continue
            d = float(np.mean(seen != g))
            if d < best:
                best = d
        # Invert: far from everything seen == novel == LOW energy (attractive).
        return 1.0 - best

    return energy


def _dist_from_start_energy(start_grid):
    """Zero-information control: distance from the START grid only.

    ADDED 2026-08-11 after adversarial review of this experiment's first run. It carries
    NOTHING the search does not already hold -- no observed corpus, no goal, no win
    state. It simply prefers states far from where the search began, which is a
    depth-first bias.

    It exists because the reviewer reproduced this experiment's entire "novelty beats
    flat" effect with exactly this control, byte-identical on bp35 and cn04. If an arm
    only matches THIS, the measured gain is a traversal-order bias and not goal search.
    A control that reproduces your effect is the most useful thing in the run.
    """
    start = np.asarray(start_grid)

    def energy(grid) -> float:
        g = np.asarray(grid)
        if g.shape != start.shape:
            return 0.0
        # Far from the start == LOW energy == attractive, matching _novelty_energy's sign.
        return 1.0 - float(np.mean(start != g))

    return energy


def _hamming_to_win_energy(win_grids):
    """Graded energy: normalised Hamming distance to the nearest KNOWN win grid.

    This is the goal-AWARE arm. It needs at least one observed level-up state, which the
    offline collector produces rarely (it restarts the episode on level-up). A game with
    no win grid is SKIPPED for this arm and the skip is recorded -- a missing measurement
    must never read as a zero.
    """
    stack = [np.asarray(g) for g in win_grids]

    def energy(grid) -> float:
        g = np.asarray(grid)
        best = 1.0
        for win in stack:
            if win.shape != g.shape:
                continue
            d = float(np.mean(win != g))
            if d < best:
                best = d
        return best

    return energy


def _run_arm(engine, goal_fn, start_grid, arm: str, energy) -> dict:
    diagnostics: dict = {}
    t0 = time.time()
    try:
        plan = e3.plan_in_model(
            engine,
            goal_fn,
            start_grid,
            max_nodes=MAX_NODES,
            goal_energy=energy,
            diagnostics=diagnostics,
        )
        err = None
    except Exception as exc:  # noqa: BLE001
        plan, err = None, repr(exc)[:200]
    return {
        "arm": arm,
        "plan_found": plan is not None,
        "plan_length": len(plan) if plan else None,
        "nodes_expanded": diagnostics.get("nodes_expanded"),
        "termination_reason": diagnostics.get("termination_reason"),
        "min_goal_energy_observed": diagnostics.get("min_goal_energy_observed"),
        "used_goal_energy_search": diagnostics.get("used_goal_energy_search"),
        "wall_s": round(time.time() - t0, 3),
        "error": err,
    }


def _run_game(game: str) -> dict:
    row: dict = {"game": game}
    source = (TRACKED_STORE / game / "world_model.py").read_text()
    engine, goal_fn = _load_engine_and_goal(source, game)
    if engine is None or goal_fn is None:
        row["error"] = "stored source lacks engine or is_level_complete"
        return row
    try:
        trans, cell = e3.collect_transitions(game, n=N_COLLECT, seed=SEED)
    except Exception as exc:  # noqa: BLE001
        row["error"] = f"collect_transitions failed: {exc!r}"[:200]
        return row
    if not trans:
        row["error"] = "no transitions collected"
        return row

    start_grid = np.asarray(trans[0].grid)
    observed = [t.grid for t in trans] + [t.next_grid for t in trans]
    win_grids = [t.next_grid for t in trans if int(t.level_after) > int(t.level_before)]
    row["n_transitions"] = len(trans)
    row["n_win_grids_observed"] = len(win_grids)

    # Is the stored predicate already true at the start? Then every arm terminates at
    # depth 0 and the game carries no information about gradients. Record, do not hide.
    try:
        row["goal_true_at_start"] = bool(goal_fn(start_grid))
    except Exception as exc:  # noqa: BLE001
        row["goal_true_at_start"] = None
        row["goal_eval_error"] = repr(exc)[:160]

    arms = {
        "flat_none": None,
        "uniform_random": make_uniform_goal_energy(SEED),
        "dist_from_start": _dist_from_start_energy(start_grid),
        "novelty": _novelty_energy(observed),
    }
    if win_grids:
        arms["hamming_to_win"] = _hamming_to_win_energy(win_grids)
    else:
        row["hamming_to_win_skipped"] = "no level-up transition observed offline"

    row["arms"] = {
        name: _run_arm(engine, goal_fn, start_grid, name, fn) for name, fn in arms.items()
    }
    return row


def _beats(candidate: dict, baseline: dict) -> bool:
    """True when the arm plans where the baseline could not, or plans more cheaply.

    CORRECTED 2026-08-11 after adversarial review. The first version justified node count
    as "the efficiency metric the scored benchmark actually rewards". That is WRONG.
    `nodes_expanded` is in-model CPU cost. The scored benchmark charges REAL ACTIONS,
    which is plan LENGTH. On ls20 the first run scored a win where the arm traded a 7-action
    plan for a 71-action plan -- a 10x regression on the metric that actually scores,
    recorded as a win on a metric that does not.

    So a cheaper search that produces a much longer plan is NOT a win. The plan-length
    guard below refuses any candidate whose plan is more than 1.5x the baseline's.
    """
    if candidate.get("error") or baseline.get("error"):
        return False
    if candidate["plan_found"] and not baseline["plan_found"]:
        return True
    if not candidate["plan_found"]:
        return False
    cl, bl = candidate.get("plan_length"), baseline.get("plan_length")
    if cl is not None and bl is not None and cl > 1.5 * bl:
        return False  # cheaper to FIND, more expensive to EXECUTE: not a win
    cn, bn = candidate.get("nodes_expanded"), baseline.get("nodes_expanded")
    if cn is None or bn is None:
        return False
    return cn < bn


def _loses(candidate: dict, baseline: dict) -> bool:
    """Explicit loss counter. The first run reported only wins, so cn04's 6.4x node
    regression was invisible in the headline. A win count without a loss count is a
    scoreboard with one team on it."""
    if candidate.get("error") or baseline.get("error"):
        return False
    if baseline["plan_found"] and not candidate["plan_found"]:
        return True
    if not candidate["plan_found"]:
        return False
    cl, bl = candidate.get("plan_length"), baseline.get("plan_length")
    if cl is not None and bl is not None and cl > 1.5 * bl:
        return True
    cn, bn = candidate.get("nodes_expanded"), baseline.get("nodes_expanded")
    if cn is None or bn is None:
        return False
    return cn > bn


def build_artifact() -> dict:
    t0 = time.time()
    roster = sorted(
        d.name
        for d in TRACKED_STORE.iterdir()
        if d.is_dir() and d.name not in NON_GAME_DIRS and (d / "world_model.py").exists()
    )
    done = json.loads(CHECKPOINT.read_text()) if CHECKPOINT.exists() else {}
    rows = list(done.get("rows", []))
    seen = {r["game"] for r in rows}
    for game in roster:
        if game in seen:
            continue
        try:
            row = _run_game(game)
        except Exception as exc:  # noqa: BLE001
            row = {"game": game, "error": repr(exc)[:200]}
        rows.append(row)
        done["rows"] = rows
        CHECKPOINT.write_text(json.dumps(done, indent=2, default=str))
        a = row.get("arms", {})
        print(
            f"[exp6252] {game}: flat={a.get('flat_none', {}).get('plan_found')} "
            f"rand={a.get('uniform_random', {}).get('plan_found')} "
            f"nov={a.get('novelty', {}).get('plan_found')} "
            f"ham={a.get('hamming_to_win', {}).get('plan_found')} "
            f"wins={row.get('n_win_grids_observed')} err={row.get('error')}",
            flush=True,
        )

    # A game is comparable only if the baseline arm actually ran and the goal was not
    # already true at the start (a depth-0 win measures nothing about gradients).
    comparable = [
        r
        for r in rows
        if not r.get("error")
        and r.get("arms", {}).get("flat_none")
        and not r["arms"]["flat_none"].get("error")
        and r.get("goal_true_at_start") is False
    ]
    graded_arms = ("uniform_random", "dist_from_start", "novelty", "hamming_to_win")
    beats_flat = {a: 0 for a in graded_arms}
    loses_flat = {a: 0 for a in graded_arms}
    arm_ran = {a: 0 for a in graded_arms}
    for r in comparable:
        base = r["arms"]["flat_none"]
        for a in graded_arms:
            cand = r["arms"].get(a)
            if not cand:
                continue
            arm_ran[a] += 1
            if _beats(cand, base):
                beats_flat[a] += 1
            if _loses(cand, base):
                loses_flat[a] += 1

    # ONLY hamming_to_win uses goal information. `novelty` is goal-BLIND by construction
    # (its own docstring says so) and the first run wrongly counted it as goal-aware, which
    # is how a goal-gradient gate was declared MET in a run where no goal signal existed.
    goal_aware = ("hamming_to_win",)
    best_goal_aware = max(goal_aware, key=lambda a: beats_flat[a]) if comparable else None
    # Two controls now, and BOTH must be cleared. `uniform_random` is information-free
    # (its grid-hash bug is fixed, so it finally reorders instead of degenerating to FIFO).
    # `dist_from_start` is the depth-first bias that reproduced the first run's whole
    # effect. An arm that only matches either one gained from traversal order, not signal.
    control_ceiling = max(beats_flat["uniform_random"], beats_flat["dist_from_start"])
    gate_met = bool(
        comparable
        and best_goal_aware
        and arm_ran[best_goal_aware] > 0
        and beats_flat[best_goal_aware] >= GATE_MIN_GAMES_BEATEN
        and beats_flat[best_goal_aware] > control_ceiling
    )

    art = {
        "experiment": "experiment_6252_goal_gradient_ab",
        "title": "Goal-gradient A/B: does any heap-key gradient beat flat BFS in the induced model?",
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "roster": roster,
        "n_games_total": len(roster),
        "n_games_comparable": len(comparable),
        "per_game_results": rows,
        "n_games_arm_beats_flat": beats_flat,
        "n_games_arm_loses_to_flat": loses_flat,
        "n_games_arm_ran": arm_ran,
        "best_goal_aware_arm": best_goal_aware,
        "gate_condition": (
            "a goal-aware arm beats flat BFS on >= 3 comparable games AND beats the "
            "uniform-random ablation count"
        ),
        "gate_min_games_beaten": GATE_MIN_GAMES_BEATEN,
        "gate_met": gate_met,
        "max_nodes_per_plan": MAX_NODES,
        "ablation_control": "uniform_random (deterministic grid hash, zero goal information)",
        "ablation_principle": (
            "a graded arm that only matches the random arm gained from search REORDERING, "
            "not from goal signal; the project's goal-induction doctrine requires this control"
        ),
        "sample_size_note": (
            "25 public games with previously-induced engines. Public games are a DEVELOPMENT "
            "PROXY for hidden-game planning, not the scored metric. A per-arm count out of "
            "~25 is a directional signal, below the project's n>=30 bar for a "
            "percentage-point claim."
        ),
        "known_limitation_win_grids": (
            "the offline collector restarts the episode on level-up, so observed win grids are "
            "rare and the hamming_to_win arm is skipped on games that produced none; a skip is "
            "recorded per game and never counted as a zero"
        ),
        "solve_provenance": "development_proxy",
        "arc_solve_claim": False,
        "verifier_is_oracle": False,
        "verifier_is_oracle_principle": (
            "the metric is planner node-efficiency inside the induced model, not the "
            "executable win-condition oracle; no level is claimed"
        ),
        "inference_substrate": "offline_arcade_live_agent_runtime_self_discovery_no_llm",
        "random_seed": SEED,
        "preconditions_checked": [
            {"resource": "tracked_e3_store_readable", "available": TRACKED_STORE.exists()},
            {"resource": "no_llm_required", "available": True},
            {"resource": "no_gpu_required", "available": True},
        ],
    }
    if not comparable:
        art["honest_verdict"] = "complete_blocked_zero_comparable_games_not_a_lever_result"
    elif gate_met:
        art["honest_verdict"] = (
            f"complete_goal_gradient_gate_met_{best_goal_aware}_beats_flat_on_"
            f"{beats_flat[best_goal_aware]}_of_{len(comparable)}_vs_random_{beats_flat['uniform_random']}"
        )
    else:
        art["honest_verdict"] = (
            f"complete_goal_gradient_gate_not_met_best_{best_goal_aware}_"
            f"{beats_flat.get(best_goal_aware or 'novelty', 0)}_of_{len(comparable)}_"
            f"vs_random_{beats_flat['uniform_random']}"
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

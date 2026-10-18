#!/usr/bin/env python3
"""REQ-ARC-WMTE-6259: does adding SENSITIVITY to the goal veto select a better candidate?

THE DECISION THIS INFORMS. The live agent sets `min_goal_predicate_consistency=1.0`, which
on a level-up-free window is pure specificity. exp6258 measured what that admits: 21 false
accepts against 5 true accepts, acceptance precision 0.1923. The obvious fix is an
ADDITIONAL condition -- accept only if the predicate ALSO fires on a known win. exp6258 shows
that would reject 21 of the 26 currently admitted. It does NOT show that doing so leaves you
better off, because rejecting a bad candidate only helps if a better one is available to
take its place. That is what this measures.

THE TWO ARMS, over the same candidate pool per game:

  * ARM A (live): accept the highest-specificity candidate. Ties broken by engine fidelity.
  * ARM B (live + sensitivity): among candidates that ALSO fire on a real win grid, accept
    the same way. If none qualifies, ARM B HAS NO CANDIDATE -- recorded explicitly, never
    silently backfilled from arm A.

THE METRIC IS END-TO-END, NOT THE GATE'S OWN OPINION. For each arm's selected candidate,
plan with `plan_in_model` and then ask whether the plan is HOLLOW: replay it inside its own
induced model and test the predicate on the terminal grid AND on the real win grid. exp6257
found four plans that terminated on an in-model false win. A gate change is only worth making
if it converts hollow plans into real ones, or converts no-plan into a real plan.

THE OUTCOME THAT WOULD KILL THE FIX, stated before the run. exp6256 found 8 of 8 freshly
induced predicates never fire on a real win. If that holds across a larger pool, ARM B will
have nothing admissible on most games, and the honest conclusion is that the gate is not the
lever -- the generator never produces a sensitive predicate, so filtering harder cannot help
and the defect is upstream in induction. `arm_b_no_admissible_candidate` makes that legible
rather than showing up as a tie.

CANDIDATE POOL. exp6256 saved two induced sources per game; those are reused rather than
re-induced. Two more are induced per game for a pool of four, so the selector has something
to choose between.

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

OUT = REPO / "results" / "experiment_6259_veto_sensitivity_ab.json"
PRIOR = REPO / "results" / "experiment_6256_win_exemplar_goal_predicate_ab.json"
CHECKPOINT = Path(
    os.environ.get("CARNOT_EXP6259_CHECKPOINT", "/tmp/carnot_exp6259_checkpoint.json")
)
ROSTER = ("dc22", "cn04", "ls20", "s5i5")
N_NEW_CANDIDATES = 2  # on top of the two reused from exp6256
N_COLLECT = 60
N_HELD = 10
MAX_NODES = 20000
SEED = 6259
GGUF_REPO_SUBSTR = "gemma-4-31B-it-qat"
SERVER_PORT = 8945
SHARED_MAX_TOKENS = 16384
INDUCE_TIMEOUT_S = 1500
LIVE_THRESHOLD = 1.0


def _frame_hud_mask(game: str):
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_competition_agent import _compute_hud_mask_from_frame

    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    return _compute_hud_mask_from_frame(env.reset())


def _logical_hud_mask(game: str, cell: int):
    m = _frame_hud_mask(game)
    return None if m is None else e3.logical_hud_mask(m, cell)


def _load_both(source: str, tag: str):
    import importlib.util
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".py", prefix=f"exp6259_{tag}_", delete=False) as f:
        f.write(source)
        path = Path(f.name)
    try:
        spec = importlib.util.spec_from_file_location(f"exp6259_{tag}_{path.stem}", path)
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        return getattr(mod, "engine", None), getattr(mod, "is_level_complete", None)
    finally:
        path.unlink(missing_ok=True)


def _score_candidate(source: str, tag: str, held, hud_mask, win) -> dict:
    """Both veto axes plus engine fidelity, for one candidate."""
    out = {
        "specificity": None,
        "fires_on_real_win": None,
        "engine_fidelity": None,
        "loadable": False,
    }
    try:
        engine, goal = _load_both(source, tag)
    except Exception as exc:  # noqa: BLE001
        out["load_error"] = repr(exc)[:160]
        return out
    out["loadable"] = engine is not None and goal is not None
    if not out["loadable"]:
        return out
    try:
        vr = e3.WorldModelVerifier(held, hud_mask=hud_mask, hud_mask_enabled=True).score(engine)
        out["engine_fidelity"] = round(float(vr.change_fidelity), 4)
    except Exception:  # noqa: BLE001
        pass
    try:
        gc = e3.score_goal_predicate_consistency(
            goal, held, engine=engine, win_grids=[win.next_grid]
        )
        out["specificity"] = round(float(gc.accuracy), 4)
        out["fires_on_real_win"] = bool(gc.sensitivity_win_grids_fired > 0)
        out["is_degenerate_constant_false"] = gc.is_degenerate_constant_false
    except Exception as exc:  # noqa: BLE001
        out["score_error"] = repr(exc)[:160]
    return out


def _plan_and_check_hollow(source: str, tag: str, start_grid, win) -> dict:
    """Plan with this candidate, then ask whether the plan is HOLLOW.

    A plan is hollow when the predicate accepts the terminal grid reached INSIDE the induced
    model but does not accept the REAL win grid -- the planner declared victory on a state
    reality does not agree with. exp6257 verified four such plans.
    """
    res = {"plan_found": False, "plan_length": None, "hollow": None, "nodes_expanded": None}
    try:
        engine, goal = _load_both(source, tag)
    except Exception:  # noqa: BLE001
        return res
    if engine is None or goal is None:
        return res
    diag: dict = {}
    try:
        plan = e3.plan_in_model(engine, goal, start_grid, max_nodes=MAX_NODES, diagnostics=diag)
    except Exception:  # noqa: BLE001
        return res
    res["nodes_expanded"] = diag.get("nodes_expanded")
    if not plan:
        return res
    res["plan_found"] = True
    res["plan_length"] = len(plan)
    try:
        g = np.asarray(start_grid).copy()
        for step in plan:
            g = np.asarray(engine(g.copy(), step["action"], step.get("data")))
        fires_terminal = bool(goal(g))
        fires_real = bool(goal(np.asarray(win.next_grid)))
        # Hollow = it ended somewhere the model calls a win while the real win state is
        # NOT accepted by the same predicate.
        res["fires_on_in_model_terminal"] = fires_terminal
        res["fires_on_real_win"] = fires_real
        res["hollow"] = bool(fires_terminal and not fires_real)
    except Exception as exc:  # noqa: BLE001
        res["replay_error"] = repr(exc)[:160]
    return res


def _select(cands: list[dict], *, require_sensitivity: bool):
    """Rank exactly as the live gate would, optionally with the extra condition.

    Live rule: admit when specificity >= 1.0. Among admitted, prefer the highest engine
    fidelity -- the reinduction loop's own retention signal -- with specificity as tiebreak.
    """
    pool = [
        c
        for c in cands
        if c["score"]["loadable"]
        and c["score"]["specificity"] is not None
        and c["score"]["specificity"] >= LIVE_THRESHOLD
    ]
    if require_sensitivity:
        pool = [c for c in pool if c["score"].get("fires_on_real_win")]
    if not pool:
        return None
    return max(
        pool,
        key=lambda c: (
            c["score"]["engine_fidelity"] if c["score"]["engine_fidelity"] is not None else -1.0,
            c["score"]["specificity"],
        ),
    )


def build_artifact() -> dict:
    t0 = time.time()
    os.environ.setdefault("CARNOT_ARC_GENERATOR_CUDA_GPU", "0,1")
    os.environ["CARNOT_ARC_GENERATOR_REQUIRE_CUDA"] = "1"
    from carnot.agentic.arc_executable_world_model import (
        GeneratorCudaRequiredError,
        LocalGGUFProposer,
    )

    reused: dict[str, list[str]] = {}
    if PRIOR.exists():
        for r in json.loads(PRIOR.read_text()).get("per_game_results", []):
            srcs = [s for s in (r.get("control_source"), r.get("treatment_source")) if s]
            if srcs:
                reused[r["game"]] = srcs

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
        train = trans[: -(2 * N_HELD)]
        held = trans[-N_HELD:]
        hud_mask = _logical_hud_mask(game, cell)
        win = e3.replay_win_transition(game, cell)
        if win is None:
            row["skipped"] = "no win grid: arm B is unconstructible and arm A has no ground truth"
            rows.append(row)
            done["rows"] = rows
            CHECKPOINT.write_text(json.dumps(done, indent=2, default=str))
            print(f"[exp6259] {game}: SKIPPED (no win grid)", flush=True)
            continue

        cands: list[dict] = []
        for i, src in enumerate(reused.get(game, [])):
            cands.append({"origin": f"reused_exp6256_{i}", "source": src})
        store = Path(os.environ["CARNOT_ARC_E3_DIR"]) / game / "world_model.py"
        for i in range(N_NEW_CANDIDATES):
            if store.exists():
                store.unlink()
            try:
                prop.induce(game, list(train), cell)
            except Exception:  # noqa: BLE001
                pass
            if store.exists():
                cands.append({"origin": f"induced_{i}", "source": store.read_text()})

        for k, c in enumerate(cands):
            c["score"] = _score_candidate(c["source"], f"{game}_{k}", held, hud_mask, win)

        start = np.asarray(trans[0].grid)
        sel_a = _select(cands, require_sensitivity=False)
        sel_b = _select(cands, require_sensitivity=True)
        row["n_candidates"] = len(cands)
        row["candidate_scores"] = [
            {"origin": c["origin"], **{k: v for k, v in c["score"].items()}} for c in cands
        ]
        row["n_admitted_by_live_veto"] = sum(
            1
            for c in cands
            if c["score"]["specificity"] is not None and c["score"]["specificity"] >= LIVE_THRESHOLD
        )
        row["n_also_sensitive"] = sum(1 for c in cands if c["score"].get("fires_on_real_win"))
        row["arm_a_selected"] = sel_a["origin"] if sel_a else None
        row["arm_b_selected"] = sel_b["origin"] if sel_b else None
        row["arm_b_no_admissible_candidate"] = sel_b is None
        if sel_a:
            row["arm_a_plan"] = _plan_and_check_hollow(sel_a["source"], f"{game}_a", start, win)
        if sel_b:
            row["arm_b_plan"] = _plan_and_check_hollow(sel_b["source"], f"{game}_b", start, win)
        rows.append(row)
        done["rows"] = rows
        CHECKPOINT.write_text(json.dumps(done, indent=2, default=str))
        print(
            f"[exp6259] {game}: {len(cands)} cands, {row['n_admitted_by_live_veto']} admitted, "
            f"{row['n_also_sensitive']} sensitive | armA={row['arm_a_selected']} "
            f"armB={row['arm_b_selected']} armB_empty={row['arm_b_no_admissible_candidate']}",
            flush=True,
        )

    comparable = [r for r in rows if r.get("n_candidates")]
    n = len(comparable)
    b_empty = [r for r in comparable if r.get("arm_b_no_admissible_candidate")]

    def _real_plan(r, arm):
        p = r.get(f"arm_{arm}_plan")
        return bool(p and p.get("plan_found") and p.get("hollow") is False)

    a_real = sum(1 for r in comparable if _real_plan(r, "a"))
    b_real = sum(1 for r in comparable if _real_plan(r, "b"))
    a_hollow = sum(
        1 for r in comparable if (r.get("arm_a_plan") or {}).get("hollow") is True
    )
    b_hollow = sum(
        1 for r in comparable if (r.get("arm_b_plan") or {}).get("hollow") is True
    )

    art = {
        "experiment": "experiment_6259_veto_sensitivity_ab",
        "title": "Does adding a sensitivity condition to the goal veto select a better candidate?",
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "roster": list(ROSTER),
        "live_threshold": LIVE_THRESHOLD,
        "per_game_results": rows,
        "n_games_comparable": n,
        "n_games_arm_b_had_no_admissible_candidate": len(b_empty),
        "arm_a_non_hollow_plans": a_real,
        "arm_b_non_hollow_plans": b_real,
        "arm_a_hollow_plans": a_hollow,
        "arm_b_hollow_plans": b_hollow,
        "games_arm_b_empty": [r["game"] for r in b_empty],
        "gate_condition": (
            "arm B produces MORE non-hollow plans than arm A. Rejecting a bad candidate only "
            "helps if a better one exists to replace it."
        ),
        "gate_met": bool(n and b_real > a_real),
        "the_outcome_that_kills_the_fix": (
            "if arm B has no admissible candidate on most games, the gate is not the lever: the "
            "generator never produces a sensitive predicate, so filtering harder cannot help and "
            "the defect is upstream in induction"
        ),
        "development_proxy_limit": (
            "win grids come from replaying banked solves through GameAdapters. A hidden game has "
            "none, and at level 1 the live agent has never won, so arm B is UNCONSTRUCTIBLE in "
            "the very case that matters most. This measures the best case for the fix."
        ),
        "sample_size_note": "at most 4 games. Directional only, far below the n>=30 bar.",
        "solve_provenance": "development_proxy",
        "arc_solve_claim": False,
        "verifier_is_oracle": False,
        "verifier_is_oracle_principle": (
            "hollowness is checked against a recorded real win grid, not against an executable "
            "oracle driving a solve; no level is claimed"
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
    elif len(b_empty) == n:
        art["honest_verdict"] = (
            f"complete_veto_sensitivity_unusable_arm_b_empty_on_all_{n}_games_gate_is_not_the_lever"
        )
    elif b_real > a_real:
        art["honest_verdict"] = (
            f"complete_veto_sensitivity_gate_met_non_hollow_{a_real}_to_{b_real}_of_{n}"
        )
    else:
        art["honest_verdict"] = (
            f"complete_veto_sensitivity_gate_not_met_non_hollow_{a_real}_to_{b_real}_of_{n}_"
            f"arm_b_empty_on_{len(b_empty)}"
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

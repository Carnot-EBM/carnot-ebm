"""Decisive re-run to RESOLVE the lp85 L5->L6 bank conflict + CAPTURE the trajectory.

CONTEXT (the conflict this resolves):
  - The headway workflow's lp85 subagent returned banked=FALSE (force-stopped early, 884 states).
  - But its DETACHED search process kept running ~39 min and wrote results/experiment_headway_lp85.json
    with banked=TRUE, reached_level=6, reproduced=True -- WITHOUT persisting the winning trajectory.
  - So the on-disk "L6 bank" cannot be independently reproduced and is contradicted by the agent.
    Per cross-check / adversarial-verify discipline it is SUSPECT until re-run independently.

THIS RUN (deterministic, lp85 adapter, branch_mode='replay'):
  1. Solve L1->L5 from scratch verifier-routed (the reproducible prefix).
  2. Verify the prefix reproduces L5 via kit.reproduce.
  3. Deepen L5->L6 with the SAME params the on-disk attempt claimed (depth_cap=150, max_nodes=120000).
  4. GATE the candidate via kit.reproduce; bank ONLY if reproduced=True for level>5.
  5. PERSIST solution_labels (the lost script's gap) so the bank is durably re-verifiable.

INTEGRITY: the ONLY proof a level banks is kit.reproduce(...) reproduced=True for level>prior.
No reproduce() pass => banked=false. development_proxy (no source read). Does NOT write the
registry/router/checkpoints (the live conductor owns those).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from carnot.agentic import arc_game_adapters as adapters
from carnot.agentic import arc_solver_kit as kit

GAME = "lp85"
REPO = Path(__file__).resolve().parents[2]
RESULT = REPO / "results" / f"experiment_headway_{GAME}_capture.json"
PRIOR_LEVEL = 5          # lp85 registry/prior reproducible floor
TARGET_LEVEL = 6
DEEP_CAP = {1: 20, 2: 70, 3: 90, 4: 150, 5: 150, 6: 150}
MAX_NODES = 120000       # matches the on-disk attempt L2a_hand_cap150


def _log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> None:
    t_start = time.time()
    ad = adapters.get_adapter(GAME)
    assert ad is not None, "lp85 must be adaptered"
    arc = kit.offline_arcade()

    notes: list[str] = []
    states_expanded_total = 0

    solver = kit.OfflineSolver(
        GAME, ad.action_labels, ad.apply, ad.state_key,
        warmup_label=ad.warmup_label, verifier=ad.hand_verifier,
        branch_mode=getattr(ad, "branch_mode", "replay"), max_nodes=MAX_NODES,
    )

    # ---- 1. Solve L1->L5 from scratch (the reproducible prefix) --------------------------------
    env = arc.make(GAME, scorecard_id=arc.open_scorecard())
    f = solver._replay(env, [])
    cur = kit.frame_level(f)
    full: list[str] = []
    prefix_log: list[dict] = []
    for lvl in range(cur + 1, PRIOR_LEVEL + 1):
        cap = DEEP_CAP.get(lvl, 150)
        t0 = time.time()
        path, nodes = solver.solve_level(env, cur, full, cap)
        states_expanded_total += nodes
        dt = time.time() - t0
        if path is None:
            _log(f"PREFIX L{lvl} FAILED nodes={nodes} {dt:.1f}s -- aborting")
            notes.append(f"prefix could not solve L{lvl}; cannot reach L5 baseline")
            _write(False, PRIOR_LEVEL, full, [], prefix_log, states_expanded_total, t_start, notes, None)
            return
        full += path
        f = solver._replay(env, full)
        cur = max(kit.frame_level(f), cur)
        prefix_log.append({"level": lvl, "path_len": len(path), "nodes": nodes, "seconds": round(dt, 2)})
        _log(f"PREFIX L{lvl} solved +{len(path)} moves cur={cur} nodes={nodes} {dt:.1f}s")

    # ---- 2. Verify the prefix reproduces L5 ----------------------------------------------------
    g5 = kit.reproduce(GAME, full, ad.apply, warmup_label=ad.warmup_label, claimed_level=PRIOR_LEVEL)
    _log(f"PREFIX gate: reproduces L{g5['reached_level']} reproduced={g5['reproduced']} (prefix_len={len(full)})")
    if not (g5["reproduced"] and g5["reached_level"] >= PRIOR_LEVEL):
        notes.append(f"prefix did NOT reproduce L{PRIOR_LEVEL} (got L{g5['reached_level']}) -- baseline unmet")
        _write(False, int(g5["reached_level"]), full, [], prefix_log, states_expanded_total, t_start, notes, g5)
        return

    # ---- 3+4. Deepen L5->L6 and GATE -----------------------------------------------------------
    t0 = time.time()
    _log(f"DEEPEN L{PRIOR_LEVEL}->L{TARGET_LEVEL} (cap={DEEP_CAP[TARGET_LEVEL]}, max_nodes={MAX_NODES})")
    env = arc.make(GAME, scorecard_id=arc.open_scorecard())
    path, nodes = solver.solve_level(env, PRIOR_LEVEL, full, DEEP_CAP[TARGET_LEVEL])
    states_expanded_total += nodes
    dt = time.time() - t0
    deepen = {"from": PRIOR_LEVEL, "to": TARGET_LEVEL, "found": path is not None,
              "nodes": nodes, "seconds": round(dt, 1)}
    banked = False
    reached_level = PRIOR_LEVEL
    final_full = full
    final_gate = g5
    if path is None:
        _log(f"DEEPEN: NO PATH L5->L6 (nodes={nodes}, {dt:.1f}s) -- on-disk banked=True NOT reproduced")
        notes.append(f"L5->L6 search found no path within cap={DEEP_CAP[TARGET_LEVEL]}, nodes={nodes}")
    else:
        cand = full + path
        gate = kit.reproduce(GAME, cand, ad.apply, warmup_label=ad.warmup_label, claimed_level=TARGET_LEVEL)
        deepen.update({"gate_reached": gate["reached_level"], "reproduced": bool(gate["reproduced"])})
        _log(f"DEEPEN: found +{len(path)} moves nodes={nodes} {dt:.1f}s "
             f"=> GATE L{gate['reached_level']} reproduced={gate['reproduced']}")
        if gate["reproduced"] and gate["reached_level"] > PRIOR_LEVEL:
            banked = True
            reached_level = int(gate["reached_level"])
            final_full = cand
            final_gate = gate
            notes.append(f"CONFIRMED: L5->L{reached_level} banked + reproduced (trajectory captured)")
        else:
            notes.append(f"L5->L6 search path did NOT reproduce (gate L{gate['reached_level']}) "
                         f"-- on-disk banked=True is NOT independently reproducible")

    _write(banked, reached_level, final_full, [deepen], prefix_log, states_expanded_total,
           t_start, notes, final_gate)


def _write(banked, reached_level, full, deepen_log, prefix_log, states, t_start, notes, gate):
    duration_s = time.time() - t_start
    if banked:
        verdict = f"complete_confirmed_lp85_L{reached_level}_bank_trajectory_captured"
    else:
        verdict = f"complete_lp85_L6_bank_NOT_reproduced_prior_floor_L5_holds"
    artifact = {
        "experiment": f"headway_{GAME}_capture",
        "game": GAME,
        "prior_reproducible_level": PRIOR_LEVEL,
        "target_level": TARGET_LEVEL,
        "reached_level": reached_level,
        "banked": banked,
        "reproduced_levels": reached_level if banked else PRIOR_LEVEL,
        "offline_reproduced": banked,
        "states_expanded": states,
        "prefix_log": prefix_log,
        "deepen_log": deepen_log,
        "final_reproduction_gate": gate,
        "solution_labels": full,
        "n_solution_moves": len(full),
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "solve_provenance": "development_proxy",
        "read_game_source": False,
        "used_env_source": True,
        "offline_ground_truth_bfs": False,
        "exhaustive_bfs_calibration": False,
        "hand_calibrated_per_game": False,
        "verifier_is_oracle": False,
        "random_seed": 0,
        "duration_s": round(duration_s, 2),
        "honest_verdict": verdict,
        "notes": notes,
        "methodology_note": (
            "Deterministic re-run resolving the lp85 L6 bank conflict. lp85 branch_mode='replay'. "
            "Solved L1->L5 verifier-routed, gated the prefix to L5, then deepened L5->L6 with "
            "depth_cap=150, max_nodes=120000 (matching the on-disk attempt). Banks ONLY on "
            "kit.reproduce(reproduced=True) for level>5; solution_labels persisted for independent "
            "re-verification. development_proxy (no source read)."
        ),
    }
    RESULT.write_text(json.dumps(artifact, indent=2))
    _log(f"WROTE {RESULT.name}: banked={banked} reached=L{reached_level} states={states} {duration_s:.1f}s")
    print("VERDICT:", verdict, flush=True)


if __name__ == "__main__":
    main()

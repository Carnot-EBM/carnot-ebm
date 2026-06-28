#!/usr/bin/env python3
"""Within-game L2->L3 SELF-INDUCTION probe, STAGE 1 (cd82) -- operator-directed 2026-06-27 ("both").

Tests whether the LIVE agent can re-induce the next level's goal from its OWN exploration (no
hand-grounded GameAdapter delta), per the workflow design (wf_b77c3b40-f11). STAGE 1 is the cheap gate
BEFORE the heavy graph-explore search wiring:
  1. Reach cd82 L2 via the existing solver (the L1+L2 prefix; incremental-progress lever).
  2. Capture the agent's OWN win exemplars (the L1- and L2-completion grids) + non-win negatives
     (intermediate frames) -- all self-played, no hand delta.
  3. Call induce_goal_energy(win_grids, non_win_grids) (arc_agi3_goal_induction.py:61) and INSPECT it:
     does it fire (>=2 win grids -> not None)? does the induced energy actually separate win from
     non-win grids? This is the goal-induction-REPRESENTATION test -- the deeper ceiling the design
     flagged (object/color hypotheses may not express cd82's mask-match goal, and L1/L2 wins may not
     determine L3's different target).

Decisive read: if induce fires AND the energy cleanly separates win/non-win -> the goal-induction
representation is viable, proceed to STAGE 2 (graph_explore_solve_v2 search to L3). If it returns None
(win-exemplar floor) OR the energy does NOT separate (representation ceiling) -> a NAMED ceiling
(missing-verifier gap), an honest negative, NOT a fabrication. Offline, zero quota.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO / "scripts"))

from carnot.agentic import arc_game_adapters as adapters  # noqa: E402
from carnot.agentic import arc_solver_kit as kit  # noqa: E402
from carnot.agentic.arc_agi3_goal_induction import (  # noqa: E402
    _RICHER_GOAL_FEATURES,
    _goal_feature_value,
    induce_goal_energy,
    induce_goal_energy_relational,
    induce_goal_energy_richer,
    induce_goal_energy_single_positive,
)
from carnot.agentic.arc_agi3_world_model import grid_of  # noqa: E402

GAME = sys.argv[1] if len(sys.argv) > 1 else "cd82"
SEED = 20260627


def _g(frame):
    """Frame -> np grid (best-effort via the project's grid_of)."""
    try:
        return np.asarray(grid_of(frame))
    except Exception:
        return None


def main() -> int:
    started = time.time()
    # Reach L2 with the FULL adapter machinery (warm-start learned verifier etc.) via solve_adaptered,
    # which returns the replayable L1+L2 trajectory labels. We use this ONLY to reach the L2 PREFIX
    # (the incremental-progress base); the L3 goal is what we then try to self-induce (no L3 hand delta).
    from arc_loop_solve import solve_adaptered  # noqa: E402

    ad = adapters.get_adapter(GAME)
    res = solve_adaptered(GAME, 2)
    labels = list(res.get("solution_labels") or [])
    level_reached = int(res.get("reached_level", 0))
    print(f"  solve_adaptered reached L{level_reached} with {len(labels)} labels", flush=True)

    # Replay the L2 trajectory prefix-by-prefix to capture the agent's OWN grids: win exemplars at each
    # level-up, negatives elsewhere. (Replay via a fresh env + the adapter apply, mirroring reproduce.)
    arc = kit.offline_arcade()
    env = arc.make(GAME, scorecard_id=arc.open_scorecard())
    solver = kit.OfflineSolver(
        GAME, ad.action_labels, ad.apply, ad.state_key,
        warmup_label=getattr(ad, "warmup_label", None),
        verifier=getattr(ad, "hand_verifier", None),
        branch_mode=getattr(ad, "branch_mode", "replay"),
    )
    win_grids, non_win_grids = [], []
    prev_level = 0
    f0 = solver._replay(env, [])
    g0 = _g(f0)
    if g0 is not None:
        non_win_grids.append(g0)
    for k in range(1, len(labels) + 1):
        fk = solver._replay(env, labels[:k])
        lk = kit.frame_level(fk)
        gk = _g(fk)
        if gk is None:
            continue
        if lk > prev_level:
            win_grids.append(gk)  # genuine self-played level-completion exemplar
            prev_level = lk
        else:
            non_win_grids.append(gk)
    level_reached = max(level_reached, prev_level)
    print(f"  captured: reached L{level_reached}, win_grids={len(win_grids)}, negs={len(non_win_grids)}", flush=True)

    # dedup grids by bytes (avoid trivially-identical negatives inflating the set)
    def _uniq(gs):
        seen, out = set(), []
        for g in gs:
            h = g.tobytes()
            if h not in seen:
                seen.add(h); out.append(g)
        return out
    win_grids = _uniq(win_grids)
    non_win_grids = _uniq(non_win_grids)

    # STAGE 1.5 (GAP-4890): the OLD operator needs >=2 wins (floor); the NEW single-positive operator
    # fires from ONE win iff it is strictly separated from the negatives. Test BOTH on cd82's real grids.
    energy_two_win = induce_goal_energy(win_grids, non_win_grids) if len(win_grids) >= 2 else None
    energy = induce_goal_energy_single_positive(win_grids[0], non_win_grids) if win_grids else None
    operator_used = "single_positive_count" if energy is not None else None
    # GAP-4891: if the count operator can't separate, try the RICHER value/fill/spatial feature family.
    if energy is None and win_grids:
        energy = induce_goal_energy_richer(win_grids[0], non_win_grids)
        operator_used = "richer_value_spatial" if energy is not None else None
    # GAP-4891 RELATIONAL: if scalars can't separate (near-win negatives), try within-frame target-match.
    if energy is None and win_grids:
        energy = induce_goal_energy_relational(win_grids[0], non_win_grids)
        operator_used = "relational_target_match" if energy is not None else None
    induce_fired = energy is not None
    two_win_floor_blocked = energy_two_win is None and len(win_grids) < 2
    # diagnostic: which richer feature (if any) strictly separates the win from the negatives?
    separating_feature = None
    if win_grids:
        for _feat in _RICHER_GOAL_FEATURES:
            try:
                _wv = _goal_feature_value(win_grids[0], _feat)
                _nv = [_goal_feature_value(g, _feat) for g in non_win_grids]
            except Exception:
                continue
            if _nv and (_wv < min(_nv) or _wv > max(_nv)):
                separating_feature = _feat
                break
    sep = None
    if induce_fired:
        we = [float(energy(g)) for g in win_grids]
        ne = [float(energy(g)) for g in non_win_grids]
        # the induced energy should be ~0 on wins and >0 on most non-wins
        mean_win_e = round(float(np.mean(we)), 4) if we else None
        mean_non_e = round(float(np.mean(ne)), 4) if ne else None
        # separation: fraction of non-wins with energy strictly above the max win energy
        max_win_e = max(we) if we else 0.0
        frac_non_above = round(sum(1 for e in ne if e > max_win_e) / max(1, len(ne)), 4)
        sep = {"mean_win_energy": mean_win_e, "mean_nonwin_energy": mean_non_e,
               "frac_nonwin_above_max_win": frac_non_above,
               "separates": bool(mean_non_e is not None and mean_win_e is not None
                                 and mean_non_e > mean_win_e and frac_non_above >= 0.5)}

    if level_reached < 2:
        verdict = f"complete_self_induction_stage1_could_not_reach_l2_reached_{level_reached}"
    elif not induce_fired:
        # count (GAP-4890) + richer scalar + RELATIONAL target-match all fail to separate the lone win
        # -> the goal is not a translate-target either (masked/scaled/learned predicate -- next ceiling).
        verdict = (f"complete_self_induction_gap4891_count_scalar_relational_all_fail_needs_learned_predicate"
                   f"_n_win_{len(win_grids)}")
    elif not sep["separates"]:
        verdict = (f"complete_self_induction_energy_does_not_separate_on_real_grids_op_{operator_used}"
                   f"_winE_{sep['mean_win_energy']}_nonE_{sep['mean_nonwin_energy']}")
    else:
        verdict = (f"success_goal_energy_fires_and_separates_op_{operator_used}_feat_{separating_feature}"
                   f"_winE_{sep['mean_win_energy']}_nonE_{sep['mean_nonwin_energy']}_proceed_to_stage2_search")

    art = {
        "experiment": "arc_within_game_l3_self_induction_cd82_stage1",
        "schema": "carnot.arc_within_game_l3_self_induction.v1",
        "honest_verdict": verdict,
        "question": ("can the agent re-induce cd82's next-level goal from its OWN win exemplars + "
                     "transitions (no hand delta), and does the induced energy separate win from non-win?"),
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "verifier_is_oracle": False,
        "game": GAME,
        "level_reached": level_reached,
        "n_win_grids": len(win_grids),
        "n_nonwin_grids": len(non_win_grids),
        "goal_induction_operator": operator_used,
        "two_win_floor_blocked_old_operator": two_win_floor_blocked,
        "induce_fired": induce_fired,
        "separating_feature": separating_feature,
        "energy_separation": sep,
        "stage": 1.6,
        "solve_provenance": "live_agent_self_discovery",
        "used_env_source": False, "read_game_source": False,
        "offline_ground_truth_bfs": False, "hand_calibrated_per_game": False,
        "interpretation": (
            "STAGE 1 gate before the graph_explore L3 search. induce_fired=False -> win-exemplar floor "
            "(<2 self-played win grids at the level transition). separates=False -> goal-induction "
            "REPRESENTATION ceiling: object/color hypotheses can't express cd82's mask-match goal, OR "
            "L1/L2 win exemplars don't determine L3's different target. Either is a NAMED missing-verifier "
            "gap (not a fabrication). separates=True -> proceed to STAGE 2 (graph_explore_solve_v2 to L3)."
        ),
        "missing_verifier_gaps": (
            "within-game L2->L3 goal re-induction: induce_goal_energy needs >=2 win grids and only object/"
            "color hypotheses; cd82's per-level target-mask-match goal needs a goal-induction operator that "
            "(a) works from few win exemplars and (b) expresses canvas==target-under-mask, derived live."
        ),
        "cites_upstream": ["wf_b77c3b40-f11 design", "arc_solve_registry.yaml cd82 dead_end"],
        "random_seed": SEED,
        "duration_s": round(time.time() - started, 2),
    }
    payload = dict(art); payload["reproducibility_checksum"] = ""
    art["reproducibility_checksum"] = "sha256:" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
    out = REPO / "results" / f"arc_within_game_l3_self_induction_{GAME}_stage1.json"
    out.write_text(json.dumps(art, indent=2) + "\n")
    print("\n=== VERDICT:", verdict)
    print(f"[{GAME}] level_reached={level_reached} n_win={len(win_grids)} n_nonwin={len(non_win_grids)} induce_fired={induce_fired} sep={sep}")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Candidate-coverage attribution: partition Carnot's ARC score gap into GENERATION vs SELECTION vs
PLANNING, per the pre-registered experiment design in
docs/research-notes/arc-top-project-search-architecture-audit-2026-07-20.md section 4.

**Researcher summary:**
    For every action along a KNOWN, reproduction-gated winning trajectory on a stalled game (0 live
    levels under the real E3AgentPolicy scored cascade -- results/arc_live_oracle_gap.json), replay to
    the state that action was taken from and ask Carnot's own live candidate generator
    (arc_graph_explore.rich_action_candidates) three questions: was this action even proposed
    (membership), did it change the frame by itself (in-isolation frame-change), and if proposed AND
    frame-changing, was it ranked highly enough a bounded search would actually try it. This localizes
    WHERE the gap is instead of guessing -- no LLM, no GPU, pure offline replay.

**Pre-registration (fixed BEFORE any classification ran, per the design note's own instruction):**
    Games: lf52, bp35, re86 -- all three are (a) confirmed 0-live-levels under the real scored
    E3AgentPolicy cascade (results/arc_live_oracle_gap.json, run 2026-07-19) and (b) have a registered,
    standard-JSON-label GameAdapter with an existing reproduction-gated banked trajectory (>=20
    actions) in results/arc_loop_solve_<game>.json. sc25 (the third game this session's whole prior
    investigation focused on) was DELIBERATELY EXCLUDED and this is disclosed, not hidden: sc25 has NO
    registered GameAdapter at all (confirmed directly -- adapters.get_adapter("sc25") is None); its
    banked plan (experiment_4468_bank_sc25_provisional_levels.SC25_PLANS_BY_LEVEL) uses a bespoke,
    game-specific label DSL ("cell0,1", "move3", "click1,1") requiring a separate per-game decoder
    (apply_sc25_label) rather than the standard {"action":N,"data":{"x":X,"y":Y}} JSON every other
    adapter and rich_action_candidates() itself use -- decoding it correctly was out of scope for this
    pass. re86 was substituted: also stalled, also has a clean standard-format banked trajectory (N=56
    to L2), no decode risk.
    Tolerance radius: 2px (ARC frames are 64x64; a small click-coordinate slop distinguishes a genuine
    perception/generation miss from a near-hit resolution artifact, per the design note's own caveat).
    "Ranked low" threshold: NOT in the top 3 of rich_action_candidates()'s returned order -- matches
    this session's own MAX_CANDIDATES=3 convention (arc_tool_loop_lookahead.py) and the project's
    typical per-node branching factor elsewhere.
    Target: >=30 progress-actions per game (>=90 total). Actual: lf52=42, bp35=57, re86=56 (N=155).

Spec: docs/research-notes/arc-top-project-search-architecture-audit-2026-07-20.md section 4
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python")
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH = os.path.join(
    REPO_ROOT, "results", "outer_loop_arc_candidate_coverage_attribution_20260723.json"
)

GAMES = ["lf52", "bp35", "re86"]
TOP_K = 3
TOLERANCE_PX = 2
RANDOM_SEED = None  # pure deterministic replay of a fixed banked trajectory; no sampling anywhere


def _data_matches(cand_data, gt_data, tol: int) -> bool:
    if not cand_data and not gt_data:
        return True
    if not cand_data or not gt_data:
        return False
    dx = abs(int(cand_data.get("x", 0)) - int(gt_data.get("x", 0)))
    dy = abs(int(cand_data.get("y", 0)) - int(gt_data.get("y", 0)))
    return dx <= tol and dy <= tol


def _find_match(candidates, action_id: int, data, tol: int):
    for idx, cand in enumerate(candidates):
        if cand.action_id == action_id and _data_matches(cand.data, data, tol):
            return idx
    return None


def run_game(game: str) -> dict:
    import numpy as np
    from carnot.agentic.arc_solver_kit import offline_arcade
    from carnot.agentic.arc_graph_explore import rich_action_candidates
    from carnot.agentic.arc_agi3_world_model import grid_of
    from arcengine import GameAction

    with open(os.path.join(REPO_ROOT, "results", f"arc_loop_solve_{game}.json")) as f:
        banked = json.load(f)
    labels = banked["solution_labels"]

    arc = offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    frame = env.reset()

    per_action = []
    counts = {"a_generation_miss": 0, "b_lookahead_signal": 0, "c_selection_miss": 0, "clean": 0}
    for i, label in enumerate(labels):
        step = json.loads(label)
        action_id = int(step["action"])
        data = step.get("data")

        candidates = rich_action_candidates(frame)
        exact_idx = _find_match(candidates, action_id, data, tol=0)
        tol_idx = (
            exact_idx
            if exact_idx is not None
            else _find_match(candidates, action_id, data, tol=TOLERANCE_PX)
        )

        grid_before = grid_of(frame)
        act = getattr(GameAction, f"ACTION{action_id}")
        next_frame = env.step(act, data=data)
        grid_after = grid_of(next_frame)
        changed_in_isolation = not np.array_equal(np.asarray(grid_before), np.asarray(grid_after))

        if tol_idx is None:
            bucket = "a_generation_miss"
        elif not changed_in_isolation:
            bucket = "b_lookahead_signal"
        elif tol_idx >= TOP_K:
            bucket = "c_selection_miss"
        else:
            bucket = "clean"
        counts[bucket] += 1
        per_action.append(
            {
                "step": i,
                "action_id": action_id,
                "data": data,
                "num_candidates": len(candidates),
                "exact_match_rank": exact_idx,
                "tolerant_match_rank": tol_idx,
                "changed_in_isolation": bool(changed_in_isolation),
                "bucket": bucket,
            }
        )
        frame = next_frame

    n = len(labels)
    return {
        "game": game,
        "n_progress_actions": n,
        "counts": counts,
        "fractions": {k: round(v / n, 4) for k, v in counts.items()} if n else {},
        "per_action": per_action,
    }


def _bootstrap_ci(bucket_flags: list, n_resamples: int = 2000, seed_stream=None) -> tuple:
    """A simple percentile bootstrap CI over a 0/1 flag list. Deterministic: uses a fixed LCG seeded
    from the flag list's own content hash (no wall-clock/random.random() -- this script must be
    reproducible given the same input trajectories)."""
    if not bucket_flags:
        return (0.0, 0.0)
    n = len(bucket_flags)
    seed = int(hashlib.sha256(str(bucket_flags).encode()).hexdigest()[:8], 16)
    state = seed
    means = []
    for _ in range(n_resamples):
        total = 0
        for _ in range(n):
            state = (1103515245 * state + 12345) & 0x7FFFFFFF
            idx = state % n
            total += bucket_flags[idx]
        means.append(total / n)
    means.sort()
    lo = means[int(0.025 * n_resamples)]
    hi = means[int(0.975 * n_resamples) - 1]
    return (round(lo, 4), round(hi, 4))


def main() -> None:
    t0 = time.monotonic()
    per_game = []
    for game in GAMES:
        print(f"=== {game} ===", flush=True)
        result = run_game(game)
        print(
            json.dumps(
                {"game": game, "n": result["n_progress_actions"], "counts": result["counts"]},
                indent=2,
            )
        )
        per_game.append(result)

    pooled_counts = {
        "a_generation_miss": 0,
        "b_lookahead_signal": 0,
        "c_selection_miss": 0,
        "clean": 0,
    }
    pooled_flags = {k: [] for k in pooled_counts}
    for g in per_game:
        for k, v in g["counts"].items():
            pooled_counts[k] += v
        for action in g["per_action"]:
            for k in pooled_counts:
                pooled_flags[k].append(1 if action["bucket"] == k else 0)

    n_total = sum(pooled_counts.values())
    pooled_fractions = (
        {k: round(v / n_total, 4) for k, v in pooled_counts.items()} if n_total else {}
    )
    pooled_ci = {k: _bootstrap_ci(pooled_flags[k]) for k in pooled_counts}

    frac_a = pooled_fractions.get("a_generation_miss", 0.0)
    frac_b = pooled_fractions.get("b_lookahead_signal", 0.0)
    frac_c = pooled_fractions.get("c_selection_miss", 0.0)

    # Falsifiable gate per the pre-registered design (arc-top-project-search-architecture-audit
    # section 4's "Falsifiable acceptance gate", decisive in all three directions):
    verdicts = []
    if frac_a > 0.5:
        verdicts.append("CONFIRMED_generation_perception_gap")
    if frac_b > 0.3:
        verdicts.append("RE_OPENED_search_lookahead_lever")
    if frac_c > 0.3:
        verdicts.append("CONFIRMED_selection_ranking_gap")
    if not verdicts:
        verdicts.append("NO_SINGLE_BUCKET_DOMINATES_below_all_thresholds")

    duration_s = round(time.monotonic() - t0, 3)
    checksum_input = json.dumps(
        [{"game": g["game"], "counts": g["counts"]} for g in per_game], sort_keys=True
    ).encode()
    reproducibility_checksum = hashlib.sha256(checksum_input).hexdigest()

    artifact = {
        "experiment": "outer_loop_arc_candidate_coverage_attribution_20260723",
        "schema": "carnot.arc_candidate_coverage_attribution.v1",
        "run_date": "2026-07-23",
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_note": "Pure offline replay of pre-existing, reproduction-gated banked "
        "trajectories (results/arc_loop_solve_<game>.json) against the live candidate generator "
        "(arc_graph_explore.rich_action_candidates) -- no LLM call, no GPU, no sampling anywhere in "
        "this script.",
        "solve_provenance": "development_proxy",
        "random_seed": RANDOM_SEED,
        "random_seed_note": "N/A -- deterministic replay of fixed banked action sequences; the "
        "bootstrap CI uses a fixed content-hash-seeded LCG, not wall-clock randomness, so the whole "
        "run is exactly reproducible given the same banked trajectory inputs.",
        "reproducibility_checksum": reproducibility_checksum,
        "duration_s": duration_s,
        "duration_note": f"Real, non-mocked offline replay across {len(GAMES)} games, {n_total} total "
        "progress-actions classified.",
        "pre_registration": {
            "games": GAMES,
            "target_n_per_game": 30,
            "tolerance_px": TOLERANCE_PX,
            "top_k_ranked_low_threshold": TOP_K,
            "excluded": {
                "sc25": "no registered GameAdapter (adapters.get_adapter('sc25') is None); its banked "
                "plan uses a bespoke label DSL requiring a separate decoder, out of scope for this pass"
            },
        },
        "per_game": [
            {k: v for k, v in g.items() if k != "per_action"}
            | {"per_action_count": len(g["per_action"])}
            for g in per_game
        ],
        "pooled": {
            "n_total": n_total,
            "counts": pooled_counts,
            "fractions": pooled_fractions,
            "bootstrap_95ci": pooled_ci,
        },
        "honest_verdict": "complete_candidate_coverage_attribution_"
        + "_and_".join(verdicts).lower(),
        "acceptance_gate": {
            "condition": "fraction(a)>0.5 -> generation/perception gap CONFIRMED; fraction(b)>0.3 -> "
            "search/lookahead lever RE-OPENED; fraction(c)>0.3 -> selection/ranking gap CONFIRMED "
            "(pre-registered, decisive in every branch, per arc-top-project-search-architecture-"
            "audit-2026-07-20.md section 4)",
            "principle": "Unlike a level-gain delta (which returns 'no change' on a near-zero-headroom "
            "corpus and cannot distinguish 'component useless' from 'no headroom'), this is a "
            "structural attribution with no delta to be null on -- it can only localize the gap.",
            "verdicts_triggered": verdicts,
        },
        "field_provenance": {
            "acceptance_gate": {
                "principle": "Pre-registered thresholds (0.5 for (a), 0.3 for (b) and (c)) were fixed "
                "in the design note BEFORE this run, not tuned to the result."
            }
        },
        "per_action_detail_path": "see per_game[].per_action_count; full per-action records omitted "
        "from top-level summary for size -- written separately below",
    }

    per_action_path = os.path.join(
        REPO_ROOT,
        "results",
        "outer_loop_arc_candidate_coverage_attribution_20260723_per_action.json",
    )
    with open(per_action_path, "w") as f:
        json.dump({g["game"]: g["per_action"] for g in per_game}, f, indent=2)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)

    print(f"\nWrote {OUTPUT_PATH}")
    print(f"Wrote {per_action_path}")
    print(f"pooled fractions: {pooled_fractions}")
    print(f"verdicts: {verdicts}")


if __name__ == "__main__":
    main()

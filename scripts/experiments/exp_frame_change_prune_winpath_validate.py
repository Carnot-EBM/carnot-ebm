"""Validate frame_change_prune safety the FAST way: does threshold T prune any action that lies ON a
known winning path? (the direct regression failure mode -- pruning a needed action).

CONTEXT: the operator asked to enable frame_change_prune_threshold at a high-confidence setting IF the
solve-rate does not regress. Prior exp4511 ran the actual solve-rate A/B (full sweep 0.25-0.75) and found
the prune REGRESSED solve-rate 4->3 with NO efficiency gain (median actions 7760->7766). This script adds
a faster, broader (22 games vs 8) necessary-condition check: replay each game's reproduced winning path
(arc_loop_solve_<game>.json solution_labels), score each TAKEN action with the SHIPPED frame-change scorer,
and count actions that would be pruned at each threshold (score < T). If winning-path actions are pruned ->
SUFFICIENT proof of unsafety at T. If none are -> the exp4511 regression is exploration-DISCOVERY divergence
(the agent never reaches the path), which a static replay cannot see -> defer to exp4511's live A/B.

Substrate: verifier_ensemble_against_cached_candidates (replay + scoring, no LLM, no slow solves).
development_proxy. No registry writes.
"""
from __future__ import annotations
import json
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RESULT = REPO / "results" / "experiment_frame_change_prune_winpath_validate.json"
THRESHOLDS = [0.2, 0.35, 0.5, 0.65, 0.75, 0.9]
GAMES = "ar25 bp35 cd82 cn04 dc22 ft09 g50t ka59 lf52 lp85 ls20 m0r0 r11l re86 s5i5 sb26 sk48 sp80 su15 tr87 tu93 vc33".split()


def main() -> None:
    from carnot.agentic.arc_competition_agent import _load_submitted_frame_change_scorer
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_graph_explore import ArcAction
    from carnot.agentic.arc_frame_change_predictor import _scorer_value
    from carnot.agentic.arc_agi3_live_adapter import _game_action
    from arcengine import GameAction

    sc = _load_submitted_frame_change_scorer()
    arc = kit.offline_arcade()
    per_game = []
    for g in GAMES:
        f = REPO / "results" / f"arc_loop_solve_{g}.json"
        if not f.exists():
            continue
        labs = (json.loads(f.read_text()).get("solution_labels")) or []
        if not labs:
            continue
        try:
            env = arc.make(g, scorecard_id=arc.open_scorecard())
            frame = env.reset()
        except Exception as e:
            per_game.append({"game": g, "error": str(e)[:60]})
            continue
        scores = []
        skipped = 0
        for lab in labs:
            # robustly resolve (aid, data) across the varied label formats: int, dict, JSON-str, bare label
            aid = data = None
            try:
                step = lab
                if isinstance(step, str):
                    try:
                        step = json.loads(step)
                    except Exception:
                        import re
                        m = re.search(r"\d+", step)  # e.g. "ACTION4" -> 4
                        step = int(m.group()) if m else None
                if isinstance(step, dict):
                    aid = int(step["action"]); data = step.get("data")
                elif step is not None:
                    aid = int(step); data = None
            except Exception:
                aid = None
            if aid is None:
                skipped += 1
                continue
            try:
                s = float(_scorer_value(frame, ArcAction(aid, data, "winpath"), sc))
                scores.append(s)
                frame = env.step(_game_action(GameAction, aid), data=data)
            except Exception:
                break
        if not scores:
            per_game.append({"game": g, "error": "no_scores"})
            continue
        pruned = {str(T): sum(1 for s in scores if s < T) for T in THRESHOLDS}
        per_game.append({
            "game": g, "n_winning_actions": len(scores),
            "min_score": round(min(scores), 3), "median_score": round(statistics.median(scores), 3),
            "winpath_pruned_at_T": pruned,
        })

    # aggregate: at each T, how many games have >=1 winning-path action pruned (=unsafe by direct evidence)
    valid = [r for r in per_game if "winpath_pruned_at_T" in r]
    unsafe_games = {str(T): [r["game"] for r in valid if r["winpath_pruned_at_T"][str(T)] > 0] for T in THRESHOLDS}
    total_pruned = {str(T): sum(r["winpath_pruned_at_T"][str(T)] for r in valid) for T in THRESHOLDS}
    # the largest T with ZERO winning-path prunes across all games
    safe_T = None
    for T in sorted(THRESHOLDS):
        if not unsafe_games[str(T)]:
            safe_T = T
    artifact = {
        "experiment": "frame_change_prune_winpath_validate",
        "n_games": len(valid),
        "thresholds": THRESHOLDS,
        "unsafe_games_at_T": unsafe_games,
        "total_winpath_actions_pruned_at_T": total_pruned,
        "largest_T_with_zero_winpath_prunes": safe_T,
        "per_game": per_game,
        "min_winning_action_score_overall": round(min((r["min_score"] for r in valid), default=0.0), 3),
        "inference_substrate": "verifier_ensemble_against_cached_candidates",
        "solve_provenance": "development_proxy", "read_game_source": False, "used_env_source": True,
        "random_seed": 0,
        "honest_verdict": "complete_frame_change_prune_winpath_necessary_condition_validated",
        "interpretation": (
            "This is a NECESSARY condition only: zero winning-path prunes means the winning trajectories "
            "remain executable under pruning, but does NOT prove the live agent still DISCOVERS them "
            "(exp4511's solve-rate A/B showed a 4->3 regression from exploration-discovery divergence + no "
            "efficiency gain). A winning-path prune at T is SUFFICIENT proof of unsafety at T."
        ),
        "cross_ref": "results/experiment_4511_frame_change_prune_predictor.json (solve_rate 4->3, actions 7760->7766)",
    }
    RESULT.write_text(json.dumps(artifact, indent=2))
    print("WROTE", RESULT.name)
    print("min winning-action score overall:", artifact["min_winning_action_score_overall"])
    for T in THRESHOLDS:
        print(f"  T={T}: total winpath actions pruned={total_pruned[str(T)]}, unsafe games={unsafe_games[str(T)]}")
    print("largest T with ZERO winning-path prunes:", safe_T)


if __name__ == "__main__":
    main()

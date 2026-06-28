#!/usr/bin/env python3
"""IGE-vs-plain-archive A/B on held-out games (operator-directed 2026-06-28 "let's try IGE-style
LLM-guided Go-Explore").

THE ONE QUESTION: does swapping the Go-Explore archive's cell-selection heuristic for an LLM
promisingness judge (Intelligent Go-Explore, arXiv:2405.15143) raise the LIVE first-win rate on held-out
games -- where the plain archive (exp4701/.433, exp4831/.445) and RND novelty (exp4688/.432) both nulled?

ISOLATED A/B (the mandatory control). BOTH arms run the identical live E3AgentPolicy with the Go-Explore
archive ENABLED on the SAME held-out color-variant specs with the SAME budget; the ONLY difference is the
selector:
  - baseline arm: archive with its built-in min(visits, -depth) heuristic (selector=None)
  - ige arm:      archive with the IGECellSelector (LLM promisingness) selector
So a positive delta is attributable to the LLM cell-judgement and nothing else. Paired by variant
signature; CI via the committed paired_first_win_delta_ci bootstrap. Reproduction-gated (kit.reproduce),
so a "win" is a real banked level, not a claim. solve_provenance=live_agent_self_discovery (the live
agent advances via its OWN exploration; the LLM only ranks the agent's own archived frontier states, it
does not hand-solve). verifier_is_oracle=False.

PRECONDITION: the IGE selector NEEDS the local Qwen3.5-9B-MTP GPU server. If it is unreachable the
selector would silently fall back to the heuristic, making both arms identical -> the A/B would be a
meaningless tie. So we hard-require the server up front and emit blocked_ige_llm_server_unreachable
otherwise (Pre-Launch Preconditions Discipline) rather than report a fake tie.

USAGE: arc_ige_llm_go_explore_ab.py [n_games] [n_variants] [budget]   (defaults tuned for a tiny slice).
The full multi-game/large-budget run is GPU-heavy and shares the dev GPU with the conductor; start small.
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

N_GAMES = int(sys.argv[1]) if len(sys.argv) > 1 else 3
N_VARIANTS = int(sys.argv[2]) if len(sys.argv) > 2 else 1
BUDGET = int(sys.argv[3]) if len(sys.argv) > 3 else 600
# Optional DFS depth-cap override (argv[4]). The submitted config caps DFS branches at max_depth=45; the
# archive only RETURNS (and thus the selector only fires) when a branch is exhausted OR hits the cap. A
# lower cap forces returns sooner so the selector is actually EXERCISED. It is applied IDENTICALLY to both
# arms (baseline heuristic + IGE), so the A/B still isolates the selector; the override is recorded so the
# result is honestly scoped ("does IGE beat the heuristic GIVEN the archive returns", not the submitted-cap
# question of whether it returns at all). Default None = submitted max_depth (45).
MAX_DEPTH = int(sys.argv[4]) if len(sys.argv) > 4 else None
# Archive cell granularity. The submitted default (bins=6) coarse-cells states so aggressively that a game
# like ar25 collapses to ~2 cells -> with the current-path exclusion there is almost never >=2 ELIGIBLE
# cells, so cell-SELECTION (IGE's entire premise) has nothing to choose among and never fires. A finer
# grid (bins~16) yields ~20+ cells so the selector has a real choice. This is itself a finding: IGE only
# matters when the archive's abstraction produces enough distinct cells. Applied to BOTH arms equally.
ARCHIVE_BINS = int(os.environ.get("CARNOT_IGE_ARCHIVE_BINS", "16"))
SEED = 20260628


def _server_reachable() -> tuple[bool, str]:
    """PRECONDITION: is the Qwen3.5-9B-MTP GPU server (or a healthy llama-server) reachable? We try the
    IGE selector's own LocalGGUFProposer._ensure_server so the check exercises the EXACT path the selector
    will use (reuses a warm server, else launches one on the dev GPU)."""
    try:
        from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

        proposer = LocalGGUFProposer(
            repo_substr="Qwen3.5-9B-MTP",
            model_path=os.environ.get("CARNOT_ARC_GGUF_PATH") or None,
            mtp=(os.environ.get("CARNOT_ARC_MTP", "1") != "0"),
            kv_quant="q8_0",
            no_think_prefix="/no_think\n",
            max_tokens=16,
            n_gpu_layers=int(os.environ.get("CARNOT_ARC_NGL", "999")),
            port=int(os.environ.get("CARNOT_IGE_LLM_PORT", "8919")),
        )
        if proposer._healthy():
            return True, "warm_server_reused"
        if proposer._ensure_server():
            return True, "server_launched"
        return False, "ensure_server_returned_false"
    except Exception as exc:  # pragma: no cover - boundary
        return False, f"exception:{exc!r}"[:160]


def main() -> int:
    started = time.time()
    from carnot.agentic.arc_ige_cell_selector import IGECellSelector
    from carnot.experiment_4605_live_integration_scored_agent import (
        _public_games,
        measurement_from_attempts,
        paired_first_win_delta_ci,
        variant_specs,
    )

    # --- PRECONDITION: LLM server reachable (else the A/B is a meaningless heuristic-vs-heuristic tie) ---
    ok, why = _server_reachable()
    preconds = [{"resource": "qwen3.5-9b-mtp_gpu_server", "available": bool(ok), "detail": why}]
    if not ok:
        art = {
            "experiment": "arc_ige_llm_go_explore_ab",
            "schema": "carnot.arc_ige_llm_go_explore_ab.v1",
            "honest_verdict": f"blocked_ige_llm_server_unreachable_{why}",
            "inference_substrate": "live_llm_inference",
            "verifier_is_oracle": False,
            "preconditions_checked": preconds,
            "solve_provenance": "live_agent_self_discovery",
            "random_seed": SEED,
            "duration_s": round(time.time() - started, 2),
        }
        _write(art)
        print("BLOCKED:", art["honest_verdict"])
        return 0

    games = _public_games(REPO)[:N_GAMES]
    variant_ids = list(range(1, N_VARIANTS + 1))
    specs = variant_specs(games, variant_ids)
    print(
        f"[ige-ab] games={games} variants={variant_ids} budget={BUDGET} specs={len(specs)}",
        flush=True,
    )

    # Shared IGE selector (one warm proposer reused across all ige-arm attempts; diagnostics accumulate).
    ige_selector = IGECellSelector()

    baseline = [
        dict(_run_attempt("baseline", str(s["game"]), s, BUDGET, None, MAX_DEPTH)) for s in specs
    ]
    ige = [
        dict(_run_attempt("ige", str(s["game"]), s, BUDGET, ige_selector, MAX_DEPTH)) for s in specs
    ]

    base_m = measurement_from_attempts(baseline)
    ige_m = measurement_from_attempts(ige)
    delta = paired_first_win_delta_ci(ige, baseline, random_seed=SEED)

    sel_diag = ige_selector.diagnostics()
    # The selector must have actually FIRED on the ige arm (else the arms are identical and the A/B is void).
    selector_exercised = int(sel_diag.get("llm_choices", 0)) > 0
    # Inference-substrate honesty (CLAUDE.md Inference-Substrate Declaration Discipline): declare
    # live_llm_inference ONLY when the LLM was genuinely invoked (the selector fired). If the archive never
    # returned >=2 eligible cells the selector made ZERO model calls, so the run is a pure live-agent search
    # over the offline arcade with NO LLM inference -- declaring live_llm_inference there would falsely trip
    # the DURATION_TOO_SHORT fabrication gate (a <60s run with a GGUF marker but no actual model call).
    substrate = (
        "live_llm_inference"
        if selector_exercised
        else "verifier_ensemble_against_cached_candidates"
    )
    model_specs = (
        {"generator": "unsloth/Qwen3.5-9B-MTP-GGUF", "kv_quant": "q8_0", "mtp": True}
        if selector_exercised
        else {"generator_declared_but_not_invoked": "unsloth/Qwen3.5-9B-MTP-GGUF", "llm_calls": 0}
    )
    ige_fw = float(ige_m["first_win_rate"])
    base_fw = float(base_m["first_win_rate"])
    ci_lo, ci_hi = (delta["ci95"][0], delta["ci95"][1])
    ige_wins = bool(ige_fw > base_fw and ci_lo > 0.0)

    if not selector_exercised:
        verdict = (
            f"complete_ige_ab_selector_not_exercised_no_archive_returns_in_{len(specs)}_specs"
            f"_budget_{BUDGET}_inconclusive"
        )
    elif ige_wins:
        verdict = (
            f"success_ige_llm_go_explore_beats_plain_archive_first_win_{ige_fw:.3f}_vs_{base_fw:.3f}"
            f"_delta_{delta['point']:.3f}_ci_excludes_0"
        )
    else:
        verdict = (
            f"complete_ige_llm_go_explore_no_first_win_lift_{ige_fw:.3f}_vs_{base_fw:.3f}"
            f"_delta_{delta['point']:.3f}_ci_{ci_lo:.3f}_{ci_hi:.3f}_generation_wall_survives"
        )

    art = {
        "experiment": "arc_ige_llm_go_explore_ab",
        "schema": "carnot.arc_ige_llm_go_explore_ab.v1",
        "honest_verdict": verdict,
        "question": (
            "does LLM-judged Go-Explore cell selection (IGE, arXiv:2405.15143) raise live first-win on "
            "held-out games vs the plain-archive heuristic (the isolated A/B; only the selector differs)?"
        ),
        "inference_substrate": substrate,
        "verifier_is_oracle": False,
        "games": games,
        "variant_ids": variant_ids,
        "budget": BUDGET,
        "max_depth_override": MAX_DEPTH,
        "archive_bins": ARCHIVE_BINS,
        "n_specs": len(specs),
        "baseline_arm": {
            k: base_m[k]
            for k in (
                "first_win_rate",
                "variant_solved_count",
                "variant_attempts_count",
                "median_actions_to_first_levelup",
            )
        },
        "ige_arm": {
            k: ige_m[k]
            for k in (
                "first_win_rate",
                "variant_solved_count",
                "variant_attempts_count",
                "median_actions_to_first_levelup",
            )
        },
        "paired_first_win_delta_ci": delta,
        "ige_selector_diagnostics": sel_diag,
        "selector_exercised": selector_exercised,
        "ige_beats_baseline": ige_wins,
        "model_specs": model_specs,
        "solve_provenance": "live_agent_self_discovery",
        "used_env_source": False,
        "read_game_source": False,
        "offline_ground_truth_bfs": False,
        "hand_calibrated_per_game": False,
        "preconditions_checked": preconds,
        "interpretation": (
            "ige_beats_baseline=True (ci excludes 0) -> LLM cell-promisingness gets winning prefixes into "
            "the explored pool where visit-count heuristics + RND novelty did not: a real first-win lever. "
            "no_first_win_lift -> the LLM judges cells fine but the winning multi-step prefix still never "
            "enters the pool: consistent with the triangulated generation/enumeration wall (.448-.452). "
            "selector_not_exercised -> budget too small for the archive to ever RETURN; raise budget."
        ),
        "cites_upstream": [
            "exp4701",
            "exp4831 (plain archive nulled)",
            "exp4688 (RND novelty nulled)",
            "arXiv:2405.15143",
        ],
        "random_seed": SEED,
        "duration_s": round(time.time() - started, 2),
    }
    _write(art)
    print("\n=== VERDICT:", verdict)
    print(
        f"baseline first_win={base_fw:.3f}  ige first_win={ige_fw:.3f}  delta={delta['point']:.3f} ci={delta['ci95']}"
    )
    print(f"selector diag: {sel_diag}")
    return 0


def _run_attempt(arm: str, game: str, spec, budget: int, selector, max_depth=None):
    """Replicates experiment_4605.run_variant_attempt's live loop, but builds the policy with the
    Go-Explore archive ALWAYS enabled and the selector being the ONLY arm difference (baseline: None;
    ige: the IGECellSelector). Reproduction-gated via kit.reproduce."""
    from arcengine import GameAction

    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_competition_agent import E3AgentPolicy
    from carnot.agentic.arc_variant_generator import VariantEnv
    from carnot.experiment_4605_live_integration_scored_agent import (
        _NoOpProposer,
        _action_label,
        _apply_action_label,
        _level_of_frame,
        _submitted_target_levels,
        _submitted_value_weight,
    )

    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    env = VariantEnv(env, game, int(spec["variant"]), reflect=spec.get("reflect"))
    archive_cfg = {"enabled": True, "bins": ARCHIVE_BINS}
    if selector is not None:
        archive_cfg["selector"] = selector
    policy = E3AgentPolicy(
        game,
        proposer=_NoOpProposer(),
        target_levels=_submitted_target_levels(),
        value_weight=_submitted_value_weight(),
        go_explore_archive=archive_cfg,
    )
    if max_depth is not None and getattr(policy, "explorer", None) is not None:
        # force the archive to RETURN sooner so the selector is exercised (applied to both arms equally)
        policy.explorer.max_depth = int(max_depth)
    frames: list = []
    latest = None
    labels: list[str] = []
    actions = 0
    start_level = None
    reached = 0
    actions_to_first = None
    for _index in range(int(budget)):
        if policy.is_done(frames, latest):
            break
        kind, data = policy.next_move(frames, latest)
        if kind == "RESET":
            latest = env.reset()
            if labels:
                labels.append("RESET")
        elif kind is None:
            break
        else:
            latest = env.step(getattr(GameAction, f"ACTION{kind}"), data=data)
            labels.append(_action_label(int(kind), data))
            actions += 1
        if start_level is None:
            start_level = _level_of_frame(latest)
        reached = _level_of_frame(latest)
        if start_level is not None and reached > start_level:
            if actions_to_first is None:
                actions_to_first = actions
            break
        frames.append(latest)
        if latest is None:
            break
    claimed = reached if start_level is not None and reached > start_level else 0
    gate = {"reproduced": False, "reached_level": 0}
    if claimed > 0 and labels:
        gate = dict(kit.reproduce(game, labels, _apply_action_label, claimed_level=claimed))
    solved = bool(gate.get("reproduced")) and int(gate.get("reached_level") or 0) >= claimed >= 1
    return {
        "game": game,
        "variant_signature": spec["variant_signature"],
        "attempted": True,
        "solved": solved,
        "first_win": solved,
        "reached_level": int(gate.get("reached_level") or reached) if solved else reached,
        "actions": actions,
        "actions_to_first_levelup": actions_to_first if solved else None,
        "reproduction_gate": gate,
        "policy_mode": arm,
    }


def _write(art: dict) -> None:
    payload = dict(art)
    payload["reproducibility_checksum"] = ""
    art["reproducibility_checksum"] = (
        "sha256:"
        + hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
    )
    out = REPO / "results" / "arc_ige_llm_go_explore_ab.json"
    out.write_text(json.dumps(art, indent=2) + "\n")
    print(f"-> {out}")


if __name__ == "__main__":
    raise SystemExit(main())

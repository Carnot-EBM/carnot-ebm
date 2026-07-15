"""Outer-loop smoke test, 2026-07-10 (extended to a multi-game suite 2026-07-15): does the
REAL LLM Strategy-Guided Exploration (SGE) mechanism (arc_llm_strategy_proposer.
SGECandidateRouter, backed by a real GPU LocalGGUFProposer) run against the offline ARC
arcade end-to-end, genuinely invoking the local model?

This is deliberately NOT a claim of a new level bank -- it is a correctness/honesty
smoke test proving `llm_strategy_proposer_used=true` with a real duration, unlike the
conductor's earlier "strategy-routed" attempt (exp5534) which never loaded a model at
all. The g50t target (prior_levels=2, target_level=3) is exp5534's own scope, kept
unchanged for a clean apples-to-apples comparison against the deterministic-template
baseline and against the REQ-ARC-FCP-5699-3 pre/post-nudge baselines.

2026-07-15 extension (operator: "can we also add more games to the sample?"): the
original single-game (g50t) run has a known, honest confound -- g50t's spawn frame
offers exactly 5 candidates with NO click action at all (see REQ-ARC-FCP-5699-2's
"honest non-bug finding"), so it cannot distinguish "the mechanism doesn't help" from
"this specific game has zero candidate-space headroom to route toward." sk48 was
already flagged in ops/known-issues.md as a richer-candidate live target (45 candidates
at spawn, action 6/click available) tried once inconclusively (no collapse triggered in
a 90-step budget). cd82 is added as a third, independent data point with its own
documented shallow L1->L2 frontier (`ops/arc_solve_registry.yaml`). Each game's
prior_levels/target_level pair matches that game's own already-precedented "shallow
frontier" framing in the registry -- not invented from scratch.

Usage: .venv/bin/python scripts/outer_loop_sge_smoke_test.py [game ...]
  No args -> runs the full GAMES suite below. One or more game ids -> runs just those.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.environ["CARNOT_ARC_DISABLE_INDUCTION"] = "1"

sys.path.insert(0, str(REPO / "python"))

from carnot.agentic import arc_solver_kit as kit  # noqa: E402
from carnot.agentic.arc_competition_agent import E3AgentPolicy, _level_of  # noqa: E402
from carnot.agentic.arc_executable_world_model import LocalGGUFProposer  # noqa: E402
from carnot.agentic.arc_llm_strategy_proposer import (  # noqa: E402
    LLMStrategyProposer,
    SGECandidateRouter,
)
from carnot.experiment_5521_arc_live_action_diverse_levelup import (  # noqa: E402
    ActionDiverseLiveGenerator,
)

# Each row: (game, prior_levels, target_level, budget). prior_levels/target_level match
# that game's own precedented shallow-frontier framing in ops/arc_solve_registry.yaml
# (not a claim this run seeds the env at that level -- see the per-game docstring notes
# below for what's actually being measured).
GAMES: tuple[tuple[str, int, int, int], ...] = (
    # exp5534's original scope, unchanged -- the established REQ-ARC-FCP-5699/5699-2/
    # 5699-3 baseline. Known confound: 5 candidates at spawn, no click action at all.
    ("g50t", 2, 3, 46),
    # Richer candidate space (45 candidates at spawn, click available) -- named in
    # ops/known-issues.md task 6 as the natural next target after g50t. Registry's own
    # "shallow_solved_sk48_L1_to_L2_live_path" framing sets this game's L1->L2 pair.
    ("sk48", 1, 2, 46),
    # Third, independent data point. Registry: "Exp4525 ... reached_level=2, banked +1
    # over the current L1 registry row" -- same L1->L2 shallow-frontier framing.
    ("cd82", 1, 2, 46),
)


class _NoOpInductionProposer:
    def induce(self, *_args, **_kwargs):
        return False, "disabled_outer_loop_sge_smoke_no_induction"

    def world_model_candidates(self, _game):
        return []


def run_game(game: str, prior_levels: int, target_level: int, budget: int, gguf) -> dict:
    proposer = LLMStrategyProposer(completer=gguf, max_tokens=64)
    router = SGECandidateRouter(
        proposer=proposer,
        game_id=game,
        k=3,
        temperatures=(0.3, 0.6, 0.9),
        max_candidates=8,
        reflect_every=6,
    )
    generator = ActionDiverseLiveGenerator(max_candidates=8)

    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    policy = E3AgentPolicy(
        game,
        proposer=_NoOpInductionProposer(),
        explore_budget=budget,
        target_levels=target_level,
        value_head=None,
        frame_change_scorer=None,
        candidate_router=router,
        action_effect_expansion_prior=False,
        action_prior=generator,
        qd_generator=generator,
        goal_bias=None,
        goal_candidate_guidance=False,
        active_probe_controller=False,
        go_explore_archive=False,
    )

    from arcengine import GameAction

    start = time.time()
    frames = []
    latest = None
    max_level = prior_levels
    action_log = []
    diagnostics_log = []
    for step in range(1, budget + 1):
        if policy.is_done(frames, latest):
            break
        before_level = int(_level_of(latest)) if latest is not None else max_level
        kind, data = policy.next_move(frames, latest)
        diag = dict(router.last_diagnostics)
        diagnostics_log.append({"step": step, **diag})
        if kind == "RESET":
            latest = env.reset()
        elif kind is None:
            break
        else:
            latest = env.step(getattr(GameAction, f"ACTION{int(kind)}"), data=data)
            after_level = int(_level_of(latest))
            max_level = max(max_level, after_level)
            router.record_outcome("level_advanced" if after_level != before_level else "no_change")
            action_log.append(
                {
                    "step": step,
                    "action": int(kind),
                    "data": dict(data) if isinstance(data, dict) else data,
                    "level_before": before_level,
                    "level_after": after_level,
                    "llm_strategy_proposer_used": diag.get("llm_strategy_proposer_used"),
                    "strategy_texts": diag.get("strategy_texts"),
                    "votes_by_index": diag.get("votes_by_index"),
                }
            )
        frames.append(latest)
        if latest is None or max_level >= target_level:
            break

    duration_s = time.time() - start
    any_llm_used = any(row.get("llm_strategy_proposer_used") for row in diagnostics_log)
    any_nudge_fired = any(row.get("reflection_nudge_fired") for row in diagnostics_log)
    return {
        "smoke_test": "outer_loop_sge_smoke_test",
        "game": game,
        "prior_levels_reproduced": prior_levels,
        "target_level": target_level,
        "max_level_reached": max_level,
        "attempts": len(action_log),
        "duration_s": duration_s,
        "llm_strategy_proposer_used_any_step": any_llm_used,
        "reflection_nudge_fired_any_step": any_nudge_fired,
        "inference_substrate": "live_llm_inference"
        if any_llm_used
        else "offline_arcade_live_agent_runtime_self_discovery_no_llm",
        "model_specs": [gguf.repo_substr],
        "action_log": action_log,
        "diagnostics_log": diagnostics_log,
    }


def main() -> int:
    requested = sys.argv[1:]
    rows = [row for row in GAMES if row[0] in requested] if requested else list(GAMES)
    if not rows:
        print(f"no matching games in {requested!r}; known games: {[g[0] for g in GAMES]}")
        return 1

    # port=8929 (not the default 8919): the default port already has a long-running HIP
    # (AMD iGPU) server on it from an unrelated process, and _ensure_server() reuses ANY
    # healthy server on the configured port regardless of which build backs it -- using a
    # fresh port forces a genuinely fresh CUDA-pinned server instead of silently inheriting
    # the slow iGPU one. GPU 1 is the outer loop's dedicated card per CLAUDE.md. One server
    # is reused SEQUENTIALLY across every game in the suite (a live healthy CUDA server on
    # a fixed port is safe and fast to reuse call-to-call within one process).
    os.environ.setdefault("CARNOT_ARC_GENERATOR_CUDA_GPU", "1")
    gguf = LocalGGUFProposer(port=8929)  # gemma-4-12B-it, GPU-enforced, fails loud

    results = []
    for game, prior_levels, target_level, budget in rows:
        print(
            f"== {game}: prior=L{prior_levels} target=L{target_level} budget={budget} ==",
            flush=True,
        )
        result = run_game(game, prior_levels, target_level, budget, gguf)
        print(
            f"   duration_s={result['duration_s']:.2f} "
            f"max_level_reached={result['max_level_reached']} "
            f"llm_used={result['llm_strategy_proposer_used_any_step']} "
            f"nudge_fired={result['reflection_nudge_fired_any_step']}",
            flush=True,
        )
        # per-game file, backward-compatible with the pre-2026-07-15 single-game path for
        # g50t specifically (prior REQ-ARC-FCP-5699-3 baselines reference this exact path).
        out_name = (
            "outer_loop_sge_smoke_test.json"
            if game == "g50t"
            else f"outer_loop_sge_smoke_test_{game}.json"
        )
        out_path = REPO / "results" / out_name
        out_path.write_text(json.dumps(result, indent=2, default=str))
        print(f"   wrote {out_path.relative_to(REPO)}", flush=True)
        results.append(result)

    summary = {
        "smoke_test_suite": "outer_loop_sge_smoke_test_suite",
        "games": [
            {
                "game": r["game"],
                "prior_levels_reproduced": r["prior_levels_reproduced"],
                "target_level": r["target_level"],
                "max_level_reached": r["max_level_reached"],
                "leveled_up": r["max_level_reached"] > r["prior_levels_reproduced"],
                "attempts": r["attempts"],
                "duration_s": r["duration_s"],
                "llm_strategy_proposer_used_any_step": r["llm_strategy_proposer_used_any_step"],
                "reflection_nudge_fired_any_step": r["reflection_nudge_fired_any_step"],
            }
            for r in results
        ],
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    summary_path = REPO / "results" / "outer_loop_sge_smoke_test_suite.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"wrote {summary_path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

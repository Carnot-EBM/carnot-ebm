"""Outer-loop smoke test, 2026-07-10: does the REAL LLM Strategy-Guided Exploration
(SGE) mechanism (arc_llm_strategy_proposer.SGECandidateRouter, backed by a real GPU
LocalGGUFProposer) run against the offline ARC arcade end-to-end, genuinely invoking
the local model?

This is deliberately NOT a claim of a new level bank -- it is a correctness/honesty
smoke test proving `llm_strategy_proposer_used=true` with a real duration, unlike the
conductor's earlier "strategy-routed" attempt (exp5534) which never loaded a model at
all. Targets the same g50t L3 frontier exp5534 already vetted via its registry
precheck (exp5533), for a clean apples-to-apples comparison against the deterministic-
template baseline.

Usage: .venv/bin/python scripts/outer_loop_sge_smoke_test.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.environ["CARNOT_ARC_DISABLE_INDUCTION"] = "1"

import sys  # noqa: E402

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


class _NoOpInductionProposer:
    def induce(self, *_args, **_kwargs):
        return False, "disabled_outer_loop_sge_smoke_no_induction"

    def world_model_candidates(self, _game):
        return []


def main() -> int:
    game = "g50t"
    prior_levels = 2
    target_level = 3
    budget = 46  # matches exp5534's scope for an honest apples-to-apples comparison

    # port=8929 (not the default 8919): the default port already has a long-running HIP
    # (AMD iGPU) server on it from an unrelated process, and _ensure_server() reuses ANY
    # healthy server on the configured port regardless of which build backs it -- using a
    # fresh port forces a genuinely fresh CUDA-pinned server instead of silently inheriting
    # the slow iGPU one. GPU 1 is the outer loop's dedicated card per CLAUDE.md.
    os.environ.setdefault("CARNOT_ARC_GENERATOR_CUDA_GPU", "1")
    gguf = LocalGGUFProposer(port=8929)  # gemma-4-12B-it, GPU-enforced, fails loud
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
    total_completer_calls = sum(
        (3 if row.get("llm_strategy_proposer_used") or row.get("completer_failure_count") else 0)
        for row in diagnostics_log
    )

    result = {
        "smoke_test": "outer_loop_sge_smoke_test",
        "game": game,
        "prior_levels_reproduced": prior_levels,
        "target_level": target_level,
        "max_level_reached": max_level,
        "attempts": len(action_log),
        "duration_s": duration_s,
        "llm_strategy_proposer_used_any_step": any_llm_used,
        "inference_substrate": "live_llm_inference" if any_llm_used else "offline_arcade_live_agent_runtime_self_discovery_no_llm",
        "model_specs": [gguf.repo_substr],
        "action_log": action_log,
        "diagnostics_log": diagnostics_log,
    }
    out_path = REPO / "results" / "outer_loop_sge_smoke_test.json"
    out_path.write_text(json.dumps(result, indent=2, default=str))
    print(f"duration_s={duration_s:.2f} max_level_reached={max_level} llm_used={any_llm_used}")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

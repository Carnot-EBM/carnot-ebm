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

CORRECTED 2026-07-15 (REQ-ARC-FCP-5699-5): `prior_levels`/`target_level` are
INFORMATIONAL labels only -- this harness has NEVER seeded the env at `prior_levels`
(no GameAdapter, no banked-trajectory replay, just a bare `env.reset()`; every game
starts at whatever level a true cold reset lands on, observed to be 0 for all 3 games
in every run to date). An earlier version of this script folded the unverified
`prior_levels` value into the SAME variable used to track the real observed level
(`max_level = prior_levels`, then `max(max_level, after_level)`), so `max_level_reached`
silently reported the assumed prior_levels forever regardless of what the run actually
did -- every prior write-up of this smoke test's results (including the original
2026-07-10 g50t run) reported an artifact of that unenforced floor, not a real
achievement. Fixed: `real_initial_level`/`real_max_level_observed`/`leveled_up` in the
returned dict are computed honestly from the actual observed level trajectory,
independent of the prior_levels/target_level labels.

2026-07-15 CONTROL added (REQ-ARC-FCP-5699-6, operator: "run it", after the corrected
finding above showed real_max_level_observed=0 on all 3 games in every SGE run):
`--baseline` swaps SGECandidateRouter for exp5534's deterministic, non-LLM
BoundedStrategyCandidateRouter under the IDENTICAL stripped-down policy config (same
budget, same disabled induction/world-model/scorer). Every run in this investigation so
far used SGE with no control -- this establishes whether ANY exploration method escapes
level 0 in this harness, or whether the harness itself (not SGE specifically) is what
caps progress.

Usage: .venv/bin/python scripts/outer_loop_sge_smoke_test.py [--baseline] [--induction] [--budget N] [game ...]
  No args -> runs the full GAMES suite below with the real SGE router, induction disabled.
  One or more game ids -> runs just those (still SGE unless --baseline is also given).
  --baseline -> runs the same games with the deterministic control router instead;
    writes to outer_loop_sge_smoke_test_baseline_<game>.json (never collides with the
    SGE-mode output paths, including g50t's unsuffixed backward-compat path).
  --induction -> re-enables the LLM world-model induction proposer (REQ-ARC-FCP-5699-8,
    a real LocalGGUFProposer/Qwen3.5-9B-MTP instead of the _NoOpInductionProposer stub)
    instead of the induction-disabled config every other run in this investigation used.
    Always writes to an _induction-suffixed path.
  --budget N -> overrides every selected game's default budget (46) with N. Always writes
    to a _budgetN-suffixed path, so a longer run never overwrites the 46-budget artifacts
    it's meant to be compared against (REQ-ARC-FCP-5699-7).
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# REQ-ARC-FCP-5699-8 (operator: "re-enable induction and run it"): CARNOT_ARC_DISABLE_INDUCTION
# is read at CALL time inside E3AgentPolicy (arc_competition_agent.py's escape hatch, "skip the
# LLM world-model induction tier entirely"), not at import time -- but it must still be set
# BEFORE run_game() constructs the policy, and --induction needs to be visible this early (before
# argparse-style parsing happens in main()) so a plain sys.argv membership check is used here
# rather than deferring to main(). Every OTHER run in this investigation left this =1 (induction
# disabled) to isolate the router under test; --induction is the one flag that flips it, letting
# a real LocalGGUFProposer (the frozen live-submission generator, same defaults E3AgentPolicy._
# proposer() would build) actually run instead of the _NoOpInductionProposer stub.
if "--induction" not in sys.argv:
    os.environ["CARNOT_ARC_DISABLE_INDUCTION"] = "1"

sys.path.insert(0, str(REPO / "python"))

from carnot.agentic import arc_solver_kit as kit  # noqa: E402
from carnot.agentic.arc_bounded_strategy_router import (  # noqa: E402
    BoundedStrategyCandidateRouter,
)
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
    # Fourth game, added 2026-07-15 (REQ-ARC-FCP-5699-9, operator: "do that", following up
    # on REQ-ARC-FCP-5699-8's finding that g50t/sk48/cd82 are ALL coincidentally members of
    # HIDDEN_STATE_GAME_IDS -- meaning induction never got a genuine chance to fire on any
    # of them, gated by select_trusted_world_model's trust_pass check before this game was
    # added. sp80 is NOT in HIDDEN_STATE_GAME_IDS (arc_world_model_trust_energy.py:22-34),
    # so a real LLM induction call should actually be attempted here from the first stall.
    # Registry: "Exp4535 ... reached_level=2, banked +1 over the current L1 registry row"
    # -- same L1->L2 shallow-frontier framing as sk48/cd82, for a clean apples-to-apples
    # comparison on the one axis that differs (hidden-state-gated vs not).
    ("sp80", 1, 2, 46),
)


class _NoOpInductionProposer:
    def induce(self, *_args, **_kwargs):
        return False, "disabled_outer_loop_sge_smoke_no_induction"

    def world_model_candidates(self, _game):
        return []


def run_game(
    game: str,
    prior_levels: int,
    target_level: int,
    budget: int,
    gguf,
    *,
    router_mode: str = "sge",
    induction_enabled: bool = False,
) -> dict:
    """router_mode="sge" (default): the real LLM Strategy-Guided Exploration router under
    test. router_mode="baseline": the CONTROL -- exp5534's deterministic, non-LLM
    BoundedStrategyCandidateRouter, under the exact same stripped-down policy config
    (same budget, same disabled induction/world-model/scorer). Added 2026-07-15 (operator:
    "run it", following the observation that every run in this investigation used SGE --
    there was no control showing whether ANY exploration method escapes level 0 in this
    harness, or whether the harness itself (not SGE specifically) is what caps progress.

    induction_enabled=False (default): every prior run in this investigation stripped LLM
    world-model induction to isolate the candidate router under test
    (_NoOpInductionProposer, a stub that always reports False/no candidates).
    induction_enabled=True (REQ-ARC-FCP-5699-8, operator: "re-enable induction and run
    it"): a real LocalGGUFProposer is constructed instead, using the SAME defaults
    E3AgentPolicy._proposer() would lazily build in production (Qwen3.5-9B-MTP, MTP, q8
    KV, /no_think) but on a dedicated port (8930) so it never collides with the SGE
    router's own gemma-4-12B-it server on 8929 -- these are two DIFFERENT models serving
    two DIFFERENT roles (induction vs. strategy proposal) and must not share a server."""
    if router_mode == "baseline":
        router = BoundedStrategyCandidateRouter(max_candidates=8)
    else:
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

    if induction_enabled:
        induction_proposer = LocalGGUFProposer(
            repo_substr="Qwen3.5-9B-MTP",
            mtp=True,
            kv_quant="q8_0",
            no_think_prefix="/no_think\n",
            max_tokens=2560,
            port=8930,
        )
    else:
        induction_proposer = _NoOpInductionProposer()

    arc = kit.offline_arcade()
    env = arc.make(game, scorecard_id=arc.open_scorecard())
    policy = E3AgentPolicy(
        game,
        proposer=induction_proposer,
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

    # HONEST LEVEL TRACKING (fixed 2026-07-15, REQ-ARC-FCP-5699-5): this harness never
    # seeds the env at `prior_levels` -- there is no GameAdapter, no banked-trajectory
    # replay, nothing before the loop but a bare `env.reset()`. Every game therefore
    # starts at whatever level a true cold reset lands on (observed to be 0 for all 3
    # games in every run to date). The ORIGINAL code initialized `max_level = prior_levels`
    # and folded real observations into that same variable via `max(max_level,
    # after_level)` -- since `prior_levels` (1 or 2) was always >= the real level ever
    # observed (0, every single run), `max_level_reached` silently reported the assumed
    # prior_levels forever, regardless of what the run actually did. Fixed: track the
    # REAL observed level honestly from the actual first frame, independent of the
    # informational prior_levels/target_level labels (which describe what OTHER solve
    # methods -- GameAdapters, banked trajectories -- have reached for this game per
    # ops/arc_solve_registry.yaml, not what this generic cold-start harness starts at).
    real_initial_level: int | None = None
    real_max_level = 0
    action_log = []
    diagnostics_log = []
    start = time.time()
    frames = []
    latest = None
    for step in range(1, budget + 1):
        if policy.is_done(frames, latest):
            break
        before_level = int(_level_of(latest)) if latest is not None else real_max_level
        kind, data = policy.next_move(frames, latest)
        diag = dict(router.last_diagnostics)
        diagnostics_log.append({"step": step, **diag})
        if kind == "RESET":
            latest = env.reset()
            if real_initial_level is None:
                real_initial_level = int(_level_of(latest))
                real_max_level = real_initial_level
        elif kind is None:
            break
        else:
            latest = env.step(getattr(GameAction, f"ACTION{int(kind)}"), data=data)
            after_level = int(_level_of(latest))
            real_max_level = max(real_max_level, after_level)
            if hasattr(router, "record_outcome"):  # BoundedStrategyCandidateRouter has no history
                router.record_outcome(
                    "level_advanced" if after_level != before_level else "no_change"
                )
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
        if latest is None or real_max_level > max(prior_levels, target_level):
            break

    duration_s = time.time() - start
    real_initial_level = real_initial_level if real_initial_level is not None else 0
    any_llm_used = any(row.get("llm_strategy_proposer_used") for row in diagnostics_log)
    any_nudge_fired = any(row.get("reflection_nudge_fired") for row in diagnostics_log)
    # Honest evidence of whether induction actually ran, not just that a real proposer was
    # configured -- policy.induction_attempts is E3AgentPolicy's own real-time log of every
    # induction attempt (planned, skipped-and-why, or genuinely invoked). A real proposer
    # with zero attempts (or every attempt skipped) means induction was never actually
    # exercised this run, which is a materially different finding from "it ran and found
    # nothing" -- report both distinctly rather than only the config flag.
    induction_attempts = list(getattr(policy, "induction_attempts", []))
    return {
        "smoke_test": "outer_loop_sge_smoke_test",
        "game": game,
        "router_mode": router_mode,
        "induction_enabled": induction_enabled,
        "induction_attempts": induction_attempts,
        "induction_attempts_not_skipped": sum(
            1 for a in induction_attempts if not a.get("skipped")
        ),
        "prior_levels_reproduced": prior_levels,
        "target_level": target_level,
        "methodology_note": (
            "prior_levels_reproduced/target_level are INFORMATIONAL labels from "
            "ops/arc_solve_registry.yaml (what OTHER solve methods reached for this game) "
            "-- this harness does NOT seed the env at that level; it always explores from "
            "a true cold env.reset(). real_initial_level/real_max_level_observed are the "
            "actual measured trajectory and are what leveled_up is computed from."
        ),
        "real_initial_level": real_initial_level,
        "real_max_level_observed": real_max_level,
        "leveled_up": real_max_level > real_initial_level,
        "max_level_reached": real_max_level,
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


class _NoLLMModelStandin:
    """model_specs stand-in for router_mode="baseline" -- no GPU/LLM is invoked at all,
    so there is nothing real to name here; this makes that explicit rather than reusing
    a GGUF proposer's repo string for a run that never touched it."""

    repo_substr = "none_deterministic_baseline_router_no_llm"


def main() -> int:
    argv = list(sys.argv[1:])
    baseline = "--baseline" in argv
    if baseline:
        argv.remove("--baseline")
    induction = "--induction" in argv
    if induction:
        argv.remove("--induction")
    budget_override: int | None = None
    if "--budget" in argv:
        idx = argv.index("--budget")
        budget_override = int(argv[idx + 1])
        del argv[idx : idx + 2]
    requested = argv
    rows = [row for row in GAMES if row[0] in requested] if requested else list(GAMES)
    if not rows:
        print(f"no matching games in {requested!r}; known games: {[g[0] for g in GAMES]}")
        return 1
    if budget_override is not None:
        rows = [(g, pl, tl, budget_override) for g, pl, tl, _ in rows]
    router_mode = "baseline" if baseline else "sge"

    if baseline:
        # CONTROL run (REQ-ARC-FCP-5699-6, operator: "run it"): BoundedStrategyCandidateRouter
        # is deterministic and invokes no LLM at all, so no GPU server is needed here.
        gguf = _NoLLMModelStandin()
    else:
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
            f"== [{router_mode}{'+induction' if induction else ''}] {game}: "
            f"prior=L{prior_levels} target=L{target_level} budget={budget} ==",
            flush=True,
        )
        result = run_game(
            game,
            prior_levels,
            target_level,
            budget,
            gguf,
            router_mode=router_mode,
            induction_enabled=induction,
        )
        print(
            f"   duration_s={result['duration_s']:.2f} "
            f"real_level={result['real_initial_level']}->{result['real_max_level_observed']} "
            f"leveled_up={result['leveled_up']} "
            f"llm_used={result['llm_strategy_proposer_used_any_step']} "
            f"nudge_fired={result['reflection_nudge_fired_any_step']} "
            f"induction_attempts_not_skipped={result['induction_attempts_not_skipped']}",
            flush=True,
        )
        # per-game file. g50t (SGE mode, default budget, induction still disabled) keeps the
        # unsuffixed pre-2026-07-15 path for backward compat with the existing REQ-ARC-FCP-
        # 5699-3/5699-4 baselines; every other combination gets an explicit, non-colliding
        # name -- a --budget or --induction flag always gets its own suffix so a differently-
        # configured run never clobbers the artifacts it's meant to be compared against.
        budget_suffix = f"_budget{budget}" if budget_override is not None else ""
        induction_suffix = "_induction" if induction else ""
        suffix = budget_suffix + induction_suffix
        if game == "g50t" and not baseline and not suffix:
            out_name = "outer_loop_sge_smoke_test.json"
        elif baseline:
            out_name = f"outer_loop_sge_smoke_test_baseline_{game}{suffix}.json"
        else:
            out_name = f"outer_loop_sge_smoke_test_{game}{suffix}.json"
        out_path = REPO / "results" / out_name
        out_path.write_text(json.dumps(result, indent=2, default=str))
        print(f"   wrote {out_path.relative_to(REPO)}", flush=True)
        results.append(result)

    summary = {
        "smoke_test_suite": "outer_loop_sge_smoke_test_suite",
        "router_mode": router_mode,
        "induction_enabled": induction,
        "games": [
            {
                "game": r["game"],
                "prior_levels_reproduced": r["prior_levels_reproduced"],
                "target_level": r["target_level"],
                "real_initial_level": r["real_initial_level"],
                "real_max_level_observed": r["real_max_level_observed"],
                "leveled_up": r["leveled_up"],
                "attempts": r["attempts"],
                "duration_s": r["duration_s"],
                "llm_strategy_proposer_used_any_step": r["llm_strategy_proposer_used_any_step"],
                "reflection_nudge_fired_any_step": r["reflection_nudge_fired_any_step"],
                "induction_attempts_not_skipped": r["induction_attempts_not_skipped"],
            }
            for r in results
        ],
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    # 2026-07-15 fix (found while testing sp80: a single-game run silently overwrote the
    # full-suite summary with just that one game's row, discarding the g50t/sk48/cd82 data
    # the file had recorded). A subset run (explicit game args, not "all GAMES") gets its
    # own suffix so it never clobbers the shared full-suite summary.
    summary_budget_suffix = f"_budget{budget_override}" if budget_override is not None else ""
    summary_induction_suffix = "_induction" if induction else ""
    summary_subset_suffix = "" if not requested else "_" + "_".join(requested)
    summary_suffix = summary_budget_suffix + summary_induction_suffix + summary_subset_suffix
    summary_name = (
        f"outer_loop_sge_smoke_test_baseline_suite{summary_suffix}.json"
        if baseline
        else f"outer_loop_sge_smoke_test_suite{summary_suffix}.json"
    )
    summary_path = REPO / "results" / summary_name
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"wrote {summary_path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

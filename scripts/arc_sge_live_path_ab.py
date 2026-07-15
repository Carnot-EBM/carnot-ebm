"""REQ-ARC-FCP-5699-12: real matched-budget A/B of SGECandidateRouter vs the SUBMITTED
discriminative router, on the ACTUAL production E3AgentPolicy config -- NOT
scripts/outer_loop_sge_smoke_test.py's deliberately-stripped harness (induction disabled,
frame-change scorer off, goal-bias off, go-explore off).

Both arms here are genuinely full-production `E3AgentPolicy(game, proposer=None)` --
`proposer=None` triggers the SAME lazy `_proposer()` real-induction-proposer default
`make_carnot_agent` uses, and every other constructor default (frame_change_scorer,
goal_bias, action_effect_expansion_prior, etc.) is left untouched, matching
`SUBMITTED_AGENT_CONFIG`. The ONLY thing that differs between arms is `candidate_router`:
BASELINE leaves it at its default (`_load_submitted_candidate_router()` ->
`CrossGameDiscriminativeCandidateRouter`); SGE arm explicitly passes a hand-built
`SGECandidateRouter` pinned to a port DIFFERENT from 8919 (the conductor's own concurrent
iGPU-backed induction proposer at the time this was written) so this measurement's
requests never queue behind or corrupt that legitimate concurrent process --
`arc_competition_agent.py`'s own `_load_sge_candidate_router()` deliberately shares the
induction proposer's port for the REAL single-process Kaggle submission, which is correct
there but wrong for A CONCURRENT DEV MEASUREMENT run sharing a GPU with the conductor.

Reuses `arc_leaderboard_eval.py`'s own `run_game()` (the real scorer:
`arc_agi.scorecard.EnvironmentScoreCalculator`, the same one the leaderboard uses) --
zero reimplementation of scoring logic.

Usage: .venv/bin/python scripts/arc_sge_live_path_ab.py <game> [--budget N]
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
os.environ.setdefault("CARNOT_ARC_GENERATOR_CUDA_GPU", "1")
sys.path.insert(0, str(REPO / "python"))

from carnot.agentic.arc_competition_agent import E3AgentPolicy  # noqa: E402
from carnot.agentic.arc_executable_world_model import LocalGGUFProposer  # noqa: E402
from carnot.agentic.arc_llm_strategy_proposer import (  # noqa: E402
    LLMStrategyProposer,
    SGECandidateRouter,
)

SGE_AB_ISOLATED_PORT = 8930  # reuses the already-warm CUDA Qwen3.5-9B-MTP server from this
# session's earlier REQ-ARC-FCP-5699-8 induction testing (avoids 8919, the conductor's own
# concurrent iGPU-backed induction proposer, and avoids risking OOM on GPU 1 -- only ~3.8GB
# free with 8929 (gemma-4-12B-it) + 8930 (this same model) already loaded).


def _load_eval_module():
    spec = importlib.util.spec_from_file_location(
        "arc_leaderboard_eval_ab", str(REPO / "scripts" / "arc_leaderboard_eval.py")
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _sge_router_isolated(game: str) -> SGECandidateRouter:
    gguf = LocalGGUFProposer(
        repo_substr="Qwen3.5-9B-MTP",
        model_path=os.environ.get("CARNOT_ARC_GGUF_PATH") or None,
        mtp=(os.environ.get("CARNOT_ARC_MTP", "1") != "0"),
        kv_quant="q8_0",
        no_think_prefix="/no_think\n",
        max_tokens=2560,
        n_gpu_layers=int(os.environ.get("CARNOT_ARC_NGL", "999")),
        port=SGE_AB_ISOLATED_PORT,
    )
    return SGECandidateRouter(
        proposer=LLMStrategyProposer(completer=gguf, max_tokens=64),
        game_id=game,
        k=3,
        temperatures=(0.3, 0.6, 0.9),
        max_candidates=8,
        reflect_every=6,
    )


def main() -> int:
    argv = sys.argv[1:]
    if not argv or argv[0].startswith("--"):
        print("usage: arc_sge_live_path_ab.py <game> [--budget N]")
        return 1
    game = argv[0]
    budget = 250
    if "--budget" in argv:
        budget = int(argv[argv.index("--budget") + 1])

    eval_mod = _load_eval_module()

    results = {}
    for arm in ("baseline_discriminative", "sge"):
        print(f"== [{arm}] {game}  budget={budget} ==", flush=True)
        if arm == "sge":
            router = _sge_router_isolated(game)
            policy = E3AgentPolicy(game, proposer=None, candidate_router=router)
        else:
            policy = E3AgentPolicy(game, proposer=None)
        candidate_router_type = type(policy.explorer.candidate_router).__name__
        start = time.time()
        result = eval_mod.run_game(game, policy, budget=budget)
        duration_s = time.time() - start
        result["candidate_router_type"] = candidate_router_type
        result["duration_s"] = duration_s
        result["arm"] = arm
        print(
            f"   duration_s={duration_s:.2f} levels={result['levels']} "
            f"reached=L{result['reached']} actions={result['actions']} "
            f"efficiency={result['efficiency']} router={candidate_router_type}",
            flush=True,
        )
        results[arm] = result

    out = {
        "experiment": "arc_sge_live_path_ab",
        "req": "REQ-ARC-FCP-5699-12",
        "game": game,
        "budget": budget,
        "methodology_note": (
            "Both arms use genuinely full-production E3AgentPolicy(game, proposer=None) -- "
            "the SAME induction/frame-change-scorer/goal-bias/action-effect-expansion-prior "
            "defaults SUBMITTED_AGENT_CONFIG ships, NOT the stripped-down outer_loop_sge_"
            "smoke_test.py harness. The only difference between arms is candidate_router. "
            "Scored via arc_leaderboard_eval.run_game() -- the real leaderboard scorer."
        ),
        "results": results,
        "run_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_path = REPO / "results" / f"arc_sge_live_path_ab_{game}.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"wrote {out_path.relative_to(REPO)}")

    base = results["baseline_discriminative"]
    sge = results["sge"]
    delta_levels = sge["levels"] - base["levels"]
    delta_eff = (sge["efficiency"] or 0.0) - (base["efficiency"] or 0.0)
    print(
        f"\n  DELTA (sge - baseline): levels={delta_levels:+d} efficiency={delta_eff:+.4f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

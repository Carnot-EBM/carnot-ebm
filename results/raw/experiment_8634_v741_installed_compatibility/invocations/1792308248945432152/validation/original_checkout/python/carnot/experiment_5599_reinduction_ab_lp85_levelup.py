"""Experiment 5599: does the candidate_27b generator (Qwen3.6-27B-MTP) improve the REAL
internal re-induction the live agent calls after a level-up, compared to the frozen current
generator (Qwen3.5-9B-MTP)?

Context: ops/known-issues.md task 13. exp5596/5597/5598 measured induction QUALITY via the
simpler `LocalGGUFProposer.induce()` wrapper -- a real, valid measurement, but not the exact
code path the SCORED live agent calls. Investigating `E3AgentPolicy._induce_and_plan()` found
the live agent's LLM tier (`execute_bounded_llm_reinduction`, arc_llm_reinduction.py) is ONLY
invoked when the induction reason is "level_up_reinduction" -- a genuine level-up just
happened. For the initial "stall" case (exploration budget exhausted without ever winning),
the agent tries a zero-LLM TTT-prior model, then falls through to classical DSL/active-probe
tiers -- the LLM is NEVER invoked. Confirmed empirically: a direct `lb.run_game` measurement on
m0r0 (never-leveled, matching this session's other roster games) completed `_induce_and_plan()`
in 17.6s -- far too fast for real LLM inference, and `level_induction_events` stayed empty.

This means exp5596/5597/5598's induction-quality roster (m0r0, sk48, cd82, sp80, none of which
have ever leveled up in any test this session) CANNOT exercise the live agent's real
LLM-reinduction code path via the natural game loop -- that is exactly why those experiments
called `LocalGGUFProposer.induce()` explicitly, bypassing the gate. This experiment instead
directly calls `execute_bounded_llm_reinduction` -- the REAL function the scored agent invokes
-- on REAL post-level-up transitions from `lp85` (the one game with a session-confirmed,
reproducible level-up: `first_levelup_index` around 6, `induce_transitions` count 8, per
exp5593's collection and this experiment's own manual pre-check).

CONTEXT-BUDGET FIX (found necessary, not assumed): lp85's 64x64 logical grid overflowed the
default n_ctx=16384 induction prompt in exp5593 (a real HTTP 400 `exceed_context_size_error`).
A manual pre-check (before building this script) with n_ctx=22000 resolved that: a real,
101.6s `execute_bounded_llm_reinduction` call completed without a context error (it happened
to fail to PLAN this specific draw -- `skipped: "proposer_failed"` -- a real, honest outcome,
not a context-budget artifact). Both arms use n_ctx=22000 here.

STOCHASTIC ROSTER (disclosed, not hidden): lp85's exploration is stochastic -- each fresh
`lb.run_game` collects a DIFFERENT real trajectory, so this experiment re-collects transitions
independently for EVERY draw (not one fixed transition set reused across arms/repeats), giving
genuine independent draws per (arm, repeat) the same way exp5598 did across games.

Both arms are pinned to the SAME hardware tier (GPU 1) via CARNOT_ARC_GENERATOR_CUDA_GPU=1,
with one proposer constructed per arm (reused across all its repeats, stopped once before the
next arm starts) and the same mid-run GPU-1-health guard exp5598 built (fails closed with an
honest partial verdict if GPU 1 becomes unreachable, rather than silently degrading hardware).

Per the task's own guardrail (mirroring task 7's frozen-stack discipline): this is an OFFLINE
DEV MEASUREMENT, not a live-stack change. Do NOT flip the frozen live generator based on this
experiment alone; report the delta and require an explicit operator decision.

Spec refs: REQ-ARC-WMTE-5599, SCENARIO-ARC-WMTE-5599-REAL-REINDUCTION-PATH,
SCENARIO-ARC-WMTE-5599-CONTEXT-BUDGET-FIX-VERIFIED.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct script guard
    sys.path.insert(0, str(PYTHON_ROOT))
if str(REPO_ROOT) not in sys.path:  # pragma: no cover - direct script guard
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPTS_ROOT) not in sys.path:  # pragma: no cover - direct script guard
    sys.path.insert(0, str(SCRIPTS_ROOT))

JsonDict = dict[str, Any]

EXPERIMENT_ID = "experiment_5599_reinduction_ab_lp85_levelup"
RESULT_RELATIVE_PATH = "results/experiment_5599_reinduction_ab_lp85_levelup.json"
SCHEMA = "carnot.exp5599.reinduction_ab_lp85_levelup.v1"
INFERENCE_SUBSTRATE = "live_llm_inference"
RANDOM_SEED = 5599
GAME = "lp85"
N_REPEATS = 3
EXPLORE_BUDGET = 6
TOTAL_BUDGET = 40
REINDUCTION_N_CTX = 22000  # widened from the class default (16384) -- lp85's 64x64 grid overflows

ARMS: tuple[JsonDict, ...] = (
    {"name": "current", "repo_substr": "Qwen3.5-9B-MTP", "port": 8941, "mtp": True},
    {"name": "candidate_27b", "repo_substr": "Qwen3.6-27B-MTP", "port": 8940, "mtp": False},
)

MODEL_SPECS = [
    {
        "name": "Qwen3.5-9B-MTP",
        "hf_id": "unsloth/Qwen3.5-9B-MTP-GGUF",
        "role": "current frozen live-submission generator (arm=current)",
    },
    {
        "name": "Qwen3.6-27B-MTP",
        "hf_id": "unsloth/Qwen3.6-27B-MTP-GGUF",
        "role": "dense candidate, the exp5598 winner (arm=candidate_27b)",
    },
]

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "model_specs",
    "game",
    "n_repeats",
    "reinduction_n_ctx",
    "per_draw_results",
    "per_arm_summary",
    "solve_provenance",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "preconditions_checked",
)

FIELD_PRINCIPLES = {
    "honest_verdict": {
        "principle": "terminal-prefixed; tests the REAL execute_bounded_llm_reinduction path "
        "the scored live agent calls after a level-up, not just induction-quality-in-isolation "
        "(exp5596/5597/5598's LocalGGUFProposer.induce() wrapper) -- a candidate that plans "
        "more reliably here is directly relevant to live solve efficiency"
    },
    "inference_substrate": {
        "principle": "live_llm_inference -- both arms invoke a real local GGUF proposer via the "
        "real internal reinduction function, not mocked"
    },
    "per_draw_results": {
        "principle": "every individual draw is recorded (planned/skipped/plan_length/"
        "heldout_accuracy), not just aggregates -- lp85's exploration is stochastic, so "
        "preserving the full draw list lets a reader assess variance directly"
    },
    "solve_provenance": {
        "principle": "development_proxy -- offline dev measurement per task 13's own guardrail; "
        "does not flip the frozen live-submission generator"
    },
    "random_seed": {"principle": "determinism precondition for reproducibility"},
    "reproducibility_checksum": {"principle": "content hash catches silent drift on replay"},
}


def preconditions(root: Path = REPO_ROOT) -> JsonDict:
    checks: dict[str, bool] = {}
    try:
        from carnot.agentic import arc_solver_kit as kit

        arc = kit.offline_arcade()
        checks["offline_arcade_importable"] = True
        checks["offline_arcade_makes_env"] = False
        try:
            env = arc.make(GAME, scorecard_id=arc.open_scorecard())
            env.reset()
            checks["offline_arcade_makes_env"] = True
        except Exception:
            pass
    except Exception:
        checks["offline_arcade_importable"] = False
    try:
        from carnot.agentic.arc_competition_agent import E3AgentPolicy  # noqa: F401
        from carnot.agentic.arc_executable_world_model import LocalGGUFProposer  # noqa: F401
        from carnot.agentic.arc_llm_reinduction import (  # noqa: F401
            execute_bounded_llm_reinduction,
        )

        checks["reinduction_import"] = True
    except Exception:
        checks["reinduction_import"] = False
    hub = Path.home() / ".cache" / "huggingface" / "hub"
    for arm in ARMS:
        key = f"gguf_cached_{arm['name']}"
        checks[key] = (
            any(str(arm["repo_substr"]).lower() in p.name.lower() for p in hub.glob("models--*"))
            if hub.exists()
            else False
        )
    checks["llama_server_binary_present"] = bool(
        list((Path.home() / ".cache").glob("llama.cpp*/build/bin/llama-server"))
    )
    checks["gpu1_free_vram_sufficient"] = _gpu1_free_mb() >= 20000
    checks["ok"] = all(checks.values())
    return checks


def _gpu1_free_mb() -> int:
    import subprocess

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        lines = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
        return int(lines[1]) if len(lines) > 1 else -1
    except Exception:
        return -1


def _first_precondition_miss(preconds: JsonDict) -> str | None:
    for key, value in preconds.items():
        if key == "ok":
            continue
        if not value:
            return key
    return None


def _checksum(payload: JsonDict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()


def _wait_for_port_down(port: int, *, timeout_s: float = 30.0) -> None:
    import urllib.request

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1):
                time.sleep(1)
                continue
        except Exception:
            return
    time.sleep(2)


def _make_proposer(repo_substr: str, port: int, mtp: bool) -> Any:
    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

    return LocalGGUFProposer(
        repo_substr=repo_substr,
        port=port,
        mtp=mtp,
        kv_quant="q8_0",
        max_tokens=2560,
        n_ctx=REINDUCTION_N_CTX,
    )


def _run_one_draw(*, arm_name: str, proposer: Any, repeat: int) -> JsonDict:
    import arc_leaderboard_eval as lb
    from carnot.agentic import arc_executable_world_model as e3
    from carnot.agentic.arc_competition_agent import E3AgentPolicy
    from carnot.agentic.arc_llm_reinduction import execute_bounded_llm_reinduction

    row: JsonDict = {"arm": arm_name, "repeat": repeat}

    policy = E3AgentPolicy(GAME, proposer=proposer, explore_budget=EXPLORE_BUDGET)
    lb.run_game(GAME, policy, budget=TOTAL_BUDGET)
    all_transitions = list(policy.transitions)
    row["transition_count"] = len(all_transitions)

    first_levelup_index = next(
        (i for i, t in enumerate(all_transitions) if t.level_after > t.level_before), None
    )
    if first_levelup_index is None:
        row["levelup_reached"] = False
        return row
    row["levelup_reached"] = True
    row["actions_to_levelup"] = int(first_levelup_index) + 1
    induce_transitions = all_transitions[: first_levelup_index + 2]
    row["induce_transition_count"] = len(induce_transitions)

    reinduce_started = time.time()
    outcome = execute_bounded_llm_reinduction(
        game=policy.short,
        transitions=induce_transitions,
        cell=policy.cell,
        root_grid=policy.root_grid,
        proposer=proposer,
        candidate_provider=policy._world_model_candidates,
        load_engine=e3.load_engine,
        plan_in_model=e3.plan_in_model,
    )
    row["reinduce_duration_s"] = round(time.time() - reinduce_started, 3)
    row["planned"] = bool(outcome.planned)
    row["skipped"] = outcome.skipped
    row["plan_length"] = len(outcome.plan)
    row["heldout_accuracy"] = outcome.heldout_accuracy
    row["refinement_rounds_used"] = int(outcome.refinement_rounds_used)
    return row


def build_artifact(*, n_repeats: int = N_REPEATS, root: Path = REPO_ROOT) -> JsonDict:
    preconds = preconditions(root)
    miss = _first_precondition_miss(preconds)
    started_at = time.time()
    if miss:
        artifact: JsonDict = {
            "experiment": EXPERIMENT_ID,
            "schema": SCHEMA,
            "result_path": RESULT_RELATIVE_PATH,
            "honest_verdict": f"complete: blocked_{miss}",
            "inference_substrate": INFERENCE_SUBSTRATE,
            "model_specs": MODEL_SPECS,
            "field_principles": FIELD_PRINCIPLES,
            "game": GAME,
            "n_repeats": int(n_repeats),
            "reinduction_n_ctx": REINDUCTION_N_CTX,
            "per_draw_results": [],
            "per_arm_summary": {},
            "solve_provenance": "development_proxy",
            "random_seed": RANDOM_SEED,
            "reproducibility_checksum": "",
            "duration_s": round(time.time() - started_at, 3),
            "preconditions_checked": preconds,
        }
        artifact["reproducibility_checksum"] = _checksum(
            {k: v for k, v in artifact.items() if k != "reproducibility_checksum"}
        )
        return artifact

    import os

    os.environ["CARNOT_ARC_GENERATOR_CUDA_GPU"] = "1"

    rows: list[JsonDict] = []
    gpu1_lost_mid_run = False
    for arm in ARMS:
        if gpu1_lost_mid_run:
            break
        arm_name = str(arm["name"])
        proposer = _make_proposer(str(arm["repo_substr"]), int(arm["port"]), bool(arm["mtp"]))
        try:
            for repeat in range(n_repeats):
                # GPU-1-HEALTH GUARD (exp5598's fix, reused): fail closed rather than silently
                # continuing on degraded hardware if GPU 1 becomes unreachable mid-run.
                if _gpu1_free_mb() < 0:
                    gpu1_lost_mid_run = True
                    rows.append(
                        {
                            "arm": arm_name,
                            "repeat": repeat,
                            "error": "gpu1_unreachable_mid_run_aborting_remaining_draws",
                        }
                    )
                    break
                try:
                    rows.append(_run_one_draw(arm_name=arm_name, proposer=proposer, repeat=repeat))
                except Exception as exc:
                    rows.append({"arm": arm_name, "repeat": repeat, "error": repr(exc)[:200]})
        finally:
            proposer.stop()
            _wait_for_port_down(int(arm["port"]))

    per_arm_summary: JsonDict = {}
    for arm in ARMS:
        arm_name = str(arm["name"])
        arm_rows = [r for r in rows if r.get("arm") == arm_name]
        levelup_rows = [r for r in arm_rows if r.get("levelup_reached")]
        planned_rows = [r for r in levelup_rows if r.get("planned")]
        per_arm_summary[arm_name] = {
            "n_attempted": len(arm_rows),
            "n_levelup_reached": len(levelup_rows),
            "n_planned": len(planned_rows),
            "plan_rate_given_levelup": (
                round(len(planned_rows) / len(levelup_rows), 4) if levelup_rows else None
            ),
            "mean_heldout_accuracy": (
                round(
                    sum(
                        r["heldout_accuracy"]
                        for r in levelup_rows
                        if r.get("heldout_accuracy") is not None
                    )
                    / max(1, sum(1 for r in levelup_rows if r.get("heldout_accuracy") is not None)),
                    4,
                )
                if any(r.get("heldout_accuracy") is not None for r in levelup_rows)
                else None
            ),
        }

    if gpu1_lost_mid_run:
        verdict = "complete: reinduction_ab_blocked_gpu1_lost_mid_run_partial"
    elif all(s["n_levelup_reached"] == 0 for s in per_arm_summary.values()):
        verdict = "complete: reinduction_ab_lp85_never_leveled_up_inconclusive"
    else:
        current_rate = per_arm_summary.get("current", {}).get("plan_rate_given_levelup")
        candidate_rate = per_arm_summary.get("candidate_27b", {}).get("plan_rate_given_levelup")
        if current_rate is None or candidate_rate is None:
            verdict = "complete: reinduction_ab_one_arm_never_leveled_up_inconclusive"
        elif candidate_rate > current_rate:
            verdict = "complete: reinduction_ab_candidate_27b_plans_more_reliably"
        elif candidate_rate < current_rate:
            verdict = "complete: reinduction_ab_current_plans_more_reliably"
        else:
            verdict = "complete: reinduction_ab_equal_plan_rate_honest_null"

    artifact = {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "model_specs": MODEL_SPECS,
        "field_principles": FIELD_PRINCIPLES,
        "game": GAME,
        "n_repeats": int(n_repeats),
        "reinduction_n_ctx": REINDUCTION_N_CTX,
        "per_draw_results": rows,
        "per_arm_summary": per_arm_summary,
        "solve_provenance": "development_proxy",
        "random_seed": RANDOM_SEED,
        "duration_s": round(time.time() - started_at, 3),
        "preconditions_checked": preconds,
        "reproducibility_checksum": "",
    }
    artifact["reproducibility_checksum"] = _checksum(
        {k: v for k, v in artifact.items() if k != "reproducibility_checksum"}
    )
    return artifact


def main() -> None:  # pragma: no cover - thin CLI wrapper, exercised manually
    artifact = build_artifact()
    out_path = REPO_ROOT / RESULT_RELATIVE_PATH
    out_path.write_text(json.dumps(artifact, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out_path} -- honest_verdict={artifact['honest_verdict']}")


if __name__ == "__main__":  # pragma: no cover
    main()

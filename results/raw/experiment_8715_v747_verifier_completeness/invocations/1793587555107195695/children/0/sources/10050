"""Bonsai-2 ternary vs. mandated Qwen3.8-27B: an ARC-domain candidate-generator eval.

Why this experiment exists (plain language).
    Two prior evals (results/experiment_bonsai2_ternary_eval.json,
    results/experiment_bonsai2_n32_collapse_followup.json,
    results/experiment_bonsai2_quality_eval.json) measured throughput and
    HumanEval code quality. None of them touched the ARC-AGI-3 live agent.
    This eval asks the ARC-specific question: if we swapped the live agent's
    generator model for the ternary model, would the LIVE decision loop (the
    scored entrypoint's own induction/action-selection cascade) behave any
    differently on real gameplay? This is OFFLINE-ONLY. It never touches the
    scored Kaggle path, the live generator pin, or the ARC solve registry.

Live entrypoint used (per CLAUDE.md "ARC Live-Path Reachability
    Discipline"): `carnot.agentic.arc_competition_agent.E3AgentPolicy`, the
    SAME class the scored Kaggle agent runs. We do not write a parallel
    solver. `game_id="unseen_runtime_game"` (an id absent from every
    registered `GameAdapter`) plus disabling the cross-game loaders (value
    head, candidate router, goal-energy bias) forces the policy onto its
    generic explorer + generic induction path -- the part of the loop a
    generator swap can actually affect. This matches, and reuses, the shape
    of the un-merged `python/carnot/experiment_10017_explorer_variants.py`
    pilot (recovered read-only via `git show explorer-pilot:...`, since that
    file was never merged to main).

Where this eval DEVIATES from experiment_10017, and why (read this before
    assuming a mistake). experiment_10017 sets
    `CARNOT_ARC_DISABLE_INDUCTION=1` for EVERY episode -- it exists to
    measure the raw explorer alone. `arc_competition_agent.py:8459` shows
    that flag skips the LLM call outright. A generator-swap eval with
    induction disabled would compare two arms that never call either model:
    a true no-op. So here induction stays ON (the default) in every arm.
    What we DO keep from experiment_10017's convention is disabling
    per-game memorized knowledge -- adapters, the cross-game value head, the
    candidate router, and goal-energy bias -- via the same
    `_disable_cross_game_loaders()` helper and the same unregistered game id,
    so a generator swap is being measured against the generic loop, not
    against one game's hand-built adapter.

Three arms, to separate two confounded variables (ternary can ONLY run
    through the PrismML fork's build; stock llama.cpp cannot read its GGUF
    format at all):
      A. today's real path: the mandated Qwen3.8-27B-Q4_K_M, served by the
         SAME binary `carnot.agentic.arc_executable_world_model._ensure_server()`
         resolves for this box with no CARNOT_LLAMA_SERVER override --
         ~/.cache/llama.cpp-master/build/bin/llama-server (stock, June 2026
         vintage, build b9606). Confirmed by reading that resolver's source,
         not assumed.
      B. fork-serving control: the SAME mandated model, served through the
         PrismML fork's llama-server instead (commit 87268f775, already
         fork-safety-audited in the first Bonsai-2 eval).
      C. the actual candidate: ternary Bonsai-2 (PTQ1_0), served through the
         same fork binary as B.
    A vs B isolates a serving-stack effect (and, honestly, ALSO a ~3.5-month
    version-vintage difference -- stock here is an old cached build, not a
    fresh same-day upstream checkout; see the report for why that is the
    correct comparator anyway: arm A is defined as "what the live agent
    actually runs today", not "generic upstream"). B vs C isolates the
    ternary-weight effect. A vs C is the bottom-line "should this even be a
    candidate" answer.

Scope and bounds (all deliberate, all disclosed, none silent):
    - GPU 1 ONLY (CUDA_VISIBLE_DEVICES pinned for the whole process before
      any subprocess is spawned). GPU 0 (the live conductor's own generator)
      is never touched.
    - Public games only, offline `environment_files` simulator, fixed seeds.
    - A SMALLER subset than the full 25 games x 3 seeds, because this eval
      runs REAL repeated generation (unlike experiment_10017's
      induction-off pilot, which needed no GPU at all and could afford
      225 episodes). See ARMS/GAMES/SEEDS below for the exact set and the
      one-line reason for each choice.
    - Induction is given a small, explicit token/timeout/context budget
      (CARNOT_ARC_INDUCE_MAX_TOKENS / _TIMEOUT / _N_CTX) so one call cannot
      consume the whole eval's wall-clock budget. This differs from the live
      agent's own much larger production budget (2400s timeout, 131072
      n_ctx) -- again, deliberate and disclosed, not a hidden confound
      shared identically across all three arms.
    - solve_provenance is ALWAYS "development_proxy": this is an outer-loop
      evaluation of a candidate generator, not a live-agent self-discovery
      session, and it does not update ops/arc_solve_registry.yaml or claim
      any new level-up.

Scoring: the real ARC scorecard formula from
    docs/research-notes/rescore-arc-formula-2026-09-26.md, reusing
    `arc_agi.scorecard.EnvironmentScoreCalculator` directly (not
    reimplemented) with baselines read from each game's downloaded
    `environment_files/<game>/*/metadata.json`. Per-level checkpoints are the
    first action index at which the live env's own `levels_completed`
    counter crosses each new level, exactly as the rescore note's own method
    section describes.

Token/timing telemetry: real numbers from real gameplay, not a synthetic
    benchmark. `LocalGGUFProposer.generate`/`.complete_text` are wrapped
    (this script only, not the shipped module) to time every call and read
    `self.last_generated_tokens` -- a field the class already populates from
    llama.cpp's own `timings.predicted_n` -- after it returns.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# GPU 1 ONLY. Must happen before any carnot import that could touch CUDA, and
# before any subprocess (llama-server) is spawned, since a child inherits this.
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "python"))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from arc_agi.scorecard import EnvironmentScoreCalculator  # noqa: E402

RESULT_PATH = REPO_ROOT / "results/experiment_bonsai2_arc_candidate_eval.json"
RAW_DIR = REPO_ROOT / "results/raw/experiment_bonsai2_arc_candidate_eval"

# ---------------------------------------------------------------------------
# environment_files resolution. This worktree does not carry the (large,
# gitignored) downloaded ARC environment corpus; it lives once in the main
# checkout. Read-only fallback, not a write target -- the "no hardcoded
# absolute write target" rule in CLAUDE.md is about paths this project
# WRITES to; this is a read of shared, pre-downloaded evaluation data.
# ---------------------------------------------------------------------------
_MAIN_CHECKOUT_ENV_FILES = Path("/home/ianblenke/github.com/ianblenke/carnot/environment_files")


def _resolve_env_dir() -> Path:
    candidates = [REPO_ROOT / "environment_files"]
    if len(REPO_ROOT.parents) >= 3:
        candidates.append(REPO_ROOT.parents[2] / "environment_files")
    candidates.append(_MAIN_CHECKOUT_ENV_FILES)
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError(f"environment_files not found in any of {candidates}")


ENV_DIR = _resolve_env_dir()


def _game_baseline_actions(game: str) -> list[int]:
    hits = sorted((ENV_DIR / game).glob("*/metadata.json"))
    if not hits:
        raise FileNotFoundError(f"no metadata.json under {ENV_DIR / game}")
    meta = json.loads(hits[0].read_text())
    baseline = meta.get("baseline_actions")
    if not isinstance(baseline, list) or not baseline:
        raise ValueError(f"{hits[0]} has no usable baseline_actions")
    return [int(v) for v in baseline]


# ---------------------------------------------------------------------------
# Arms. Ports are distinct per arm so the proposer's own port-reuse check
# (which compares model path + n_ctx, NOT server binary) can never mistake
# one arm's server for another's.
# ---------------------------------------------------------------------------
STOCK_LLAMA_SERVER = Path.home() / ".cache/llama.cpp-master/build/bin/llama-server"
FORK_LLAMA_SERVER = Path("/tmp/bonsai_followup/fork-llama.cpp/build/bin/llama-server")
GGUF_STANDARD = (
    Path.home() / ".cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF/snapshots/"
    "fe1e2a23d973adb629709749dc4f6756df66ef10/Qwen3.8-27B-Q4_K_M.gguf"
)
GGUF_TERNARY = (
    Path.home() / ".cache/huggingface/hub/models--prism-ml--Ternary-Bonsai-2-27B-gguf/snapshots/"
    "b072e1d3b35a0a630cece372c2127528e0994386/Ternary-Bonsai-2-27B-PTQ1_0.gguf"
)

ARMS: dict[str, dict[str, Any]] = {
    "A_stock_standard": {
        "server": STOCK_LLAMA_SERVER,
        "gguf": GGUF_STANDARD,
        "repo": "unsloth/Qwen3.8-27B-GGUF",
        "port": 18919,
        "label": "today's real path: stock llama.cpp-master build, mandated Q4_K_M model",
    },
    "B_fork_standard": {
        "server": FORK_LLAMA_SERVER,
        "gguf": GGUF_STANDARD,
        "repo": "unsloth/Qwen3.8-27B-GGUF",
        "port": 18920,
        "label": "fork-serving control: PrismML fork build, SAME mandated model",
    },
    "C_fork_ternary": {
        "server": FORK_LLAMA_SERVER,
        "gguf": GGUF_TERNARY,
        "repo": "prism-ml/Ternary-Bonsai-2-27B-gguf",
        "port": 18921,
        "label": "the candidate: PrismML fork build, ternary Bonsai-2 (PTQ1_0)",
    },
}

# Bounded induction budget -- see module docstring "Scope and bounds".
INDUCE_MAX_TOKENS = 512
INDUCE_TIMEOUT_S = 90
INDUCE_N_CTX = 16384

SEEDS = (7491001, 7491002)
# Six games: five with recorded non-zero V0 signal in the rescore study (a
# generic-loop win is at least possible), one (ar25) with recorded zero
# signal as a negative control (an arm should not "invent" a win there).
GAMES = ("cd82", "lp85", "tu93", "vc33", "sp80", "ar25")
ACTION_BUDGET = 300
# 320s, not 240s: measured directly (isolation diagnostic, cd82 seed 7491002,
# stock binary) that one real induce call took 53.6s at the 512-token budget
# and the cascade retries up to 3 times, so one stall event alone can cost
# up to ~270s. 320s gives room for one full retry cycle plus explore overhead
# without letting a single stuck episode consume the whole eval's budget.
EPISODE_WALL_CAP_S = 320


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _checksum(value: Any) -> str:
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _progress(msg: str) -> None:
    print(f"[bonsai2-arc {time.strftime('%H:%M:%S')}] {msg}", flush=True)


class EpisodeTimeout(Exception):
    pass


def _make_proposer(arm_key: str):
    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

    cfg = ARMS[arm_key]
    return LocalGGUFProposer(
        model_path=str(cfg["gguf"]),
        model_repository=cfg["repo"],
        model_filename=cfg["gguf"].name,
        port=cfg["port"],
        n_ctx=INDUCE_N_CTX,
        max_tokens=INDUCE_MAX_TOKENS,
        timeout=INDUCE_TIMEOUT_S,
        no_think_prefix="/no_think\n",
        kv_quant="q8_0",
        n_gpu_layers=999,
        mtp=False,
        ffn_cpu_layers=0,
    )


def _install_instrumentation():
    """Wrap LocalGGUFProposer.generate/complete_text to time every real call and
    read llama.cpp's own reported token count. Returns (call_log, restore_fn).
    This patches ONLY this script's in-process view of the class; it is never
    committed into arc_executable_world_model.py.
    """

    import carnot.agentic.arc_executable_world_model as e3mod

    call_log: list[dict[str, Any]] = []
    orig_generate = e3mod.LocalGGUFProposer.generate
    orig_complete_text = e3mod.LocalGGUFProposer.complete_text

    def timed_generate(self, *args, **kwargs):
        t0 = time.monotonic()
        result = orig_generate(self, *args, **kwargs)
        dt = time.monotonic() - t0
        call_log.append(
            {
                "method": "generate",
                "duration_s": round(dt, 4),
                "tokens": int(getattr(self, "last_generated_tokens", -1)),
                "ok": bool(result[0]),
            }
        )
        return result

    def timed_complete_text(self, *args, **kwargs):
        t0 = time.monotonic()
        result = orig_complete_text(self, *args, **kwargs)
        dt = time.monotonic() - t0
        call_log.append(
            {
                "method": "complete_text",
                "duration_s": round(dt, 4),
                "tokens": int(getattr(self, "last_generated_tokens", -1)),
                "ok": bool(result[0]),
            }
        )
        return result

    e3mod.LocalGGUFProposer.generate = timed_generate
    e3mod.LocalGGUFProposer.complete_text = timed_complete_text

    def restore() -> None:
        e3mod.LocalGGUFProposer.generate = orig_generate
        e3mod.LocalGGUFProposer.complete_text = orig_complete_text

    return call_log, restore


def run_episode(
    proposer,
    game: str,
    seed: int,
    call_log: list[dict[str, Any]],
    *,
    budget: int = ACTION_BUDGET,
    wall_cap_s: int = EPISODE_WALL_CAP_S,
) -> dict[str, Any]:
    """One episode of the LIVE scored decision loop (E3AgentPolicy) against the
    offline arcade, generic-loop-only (adapter/cross-game knowledge disabled),
    with the given arm's proposer doing all real generation.
    """

    from arcengine import GameAction
    from carnot.agentic import arc_executable_world_model as e3
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_agi3_world_model import frame_hash, grid_of
    from carnot.agentic.arc_competition_agent import E3AgentPolicy
    from carnot.experiment_7471_v654_arc_seam_observation import (
        _disable_cross_game_loaders,
        _restore_cross_game_loaders,
    )

    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    os.environ["CARNOT_ARC_RANDOM_SEED"] = str(seed)
    os.environ["CARNOT_ARC_GENERATOR_SEED"] = str(seed)
    # induction is left ON (no CARNOT_ARC_DISABLE_INDUCTION) -- see module docstring.
    kit.ENV_DIR = ENV_DIR
    old_e3_dir = e3.E3_DIR
    episode_id = f"{game}__seed-{seed}"
    e3.E3_DIR = RAW_DIR / "empty_engine_store" / episode_id

    def alarm_handler(_signum: int, _frame: Any) -> None:
        raise EpisodeTimeout(f"episode exceeded {wall_cap_s}s wall-clock")

    prev_alarm = signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(wall_cap_s)
    calls_before = len(call_log)
    t0 = time.monotonic()
    originals = _disable_cross_game_loaders()
    error: str | None = None
    peak_level = 0
    level_checkpoints: dict[int, int] = {}
    final_actions = 0
    try:
        policy = E3AgentPolicy(
            "unseen_runtime_game", proposer=proposer, frontier_discipline_seed=seed
        )
    finally:
        _restore_cross_game_loaders(originals)
    arcade = kit.offline_arcade()
    env = arcade.make(game, scorecard_id=arcade.open_scorecard())
    frames: list[Any] = []
    latest: Any = None
    last_phase = None
    debug = os.environ.get("BONSAI2_ARC_DEBUG") == "1"
    try:
        for index in range(1, budget + 1):
            if policy.is_done(frames, latest):
                break
            if debug and getattr(policy, "phase", None) != last_phase:
                last_phase = policy.phase
                _progress(f"    action {index}: phase -> {last_phase}")
            t_action = time.monotonic()
            kind, data = policy.next_move(frames, latest)
            action_dt = time.monotonic() - t_action
            if debug and action_dt > 2.0:
                _progress(f"    action {index} took {action_dt:.1f}s (phase={policy.phase})")
            if kind is None:
                break
            if kind == "RESET":
                latest = env.reset()
            else:
                latest = env.step(getattr(GameAction, f"ACTION{kind}"), data=data)
            policy.observe_action_outcome(latest)
            frames.append(latest)
            level = int(getattr(latest, "levels_completed", 0) or 0)
            if level > peak_level:
                for lv in range(peak_level + 1, level + 1):
                    level_checkpoints[lv] = index
                peak_level = level
            final_actions = index
            _ = frame_hash(grid_of(latest))  # sanity: frame decodes; not stored per-action here
    except EpisodeTimeout as exc:
        error = str(exc)
    except Exception as exc:  # noqa: BLE001 - a real-env episode must not crash the eval
        error = f"{type(exc).__name__}: {exc}"[:500]
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev_alarm)
        e3.E3_DIR = old_e3_dir
    duration_s = round(time.monotonic() - t0, 3)
    episode_calls = call_log[calls_before:]
    return {
        "game": game,
        "seed": seed,
        "action_budget": budget,
        "actions_taken": final_actions,
        "levels_completed": peak_level,
        "level_checkpoints": {str(k): v for k, v in level_checkpoints.items()},
        "error": error,
        "duration_s": duration_s,
        "llm_calls": episode_calls,
        "llm_call_count": len(episode_calls),
        "llm_tokens_total": sum(c["tokens"] for c in episode_calls if c["tokens"] > 0),
        "llm_generation_duration_s": round(sum(c["duration_s"] for c in episode_calls), 3),
        "adapter_disabled": True,
        "banked_trajectories_disabled": True,
        "stored_engines_disabled": True,
        "induction_enabled": True,
    }


def real_scorecard_score(baseline_actions: list[int], row: dict[str, Any]) -> dict[str, Any]:
    """Rescore one episode with the REAL formula, reusing
    arc_agi.scorecard.EnvironmentScoreCalculator directly (see
    docs/research-notes/rescore-arc-formula-2026-09-26.md for the method this
    reproduces: iterate every level in the game's baseline vector, mark
    completed=True up to the peak level reached, completed=False (score 0)
    for the first incomplete level and every level after it).
    """

    checkpoints = {int(k): v for k, v in row["level_checkpoints"].items()}
    peak = max(checkpoints.keys(), default=0)
    calc = EnvironmentScoreCalculator()
    prev = 0
    for level_idx in range(1, len(baseline_actions) + 1):
        baseline = baseline_actions[level_idx - 1]
        if level_idx <= peak:
            ck = checkpoints[level_idx]
            calc.add_level(
                level_index=level_idx,
                completed=True,
                actions_taken=ck - prev,
                baseline_actions=baseline,
            )
            prev = ck
        elif level_idx == peak + 1:
            actions_taken = row["actions_taken"] - prev
            calc.add_level(
                level_index=level_idx,
                completed=False,
                actions_taken=actions_taken,
                baseline_actions=baseline,
            )
            prev = row["actions_taken"]
        else:
            calc.add_level(
                level_index=level_idx, completed=False, actions_taken=0, baseline_actions=baseline
            )
    result = calc.to_score(include_levels=True)
    return {"game_score": result.score, "per_level_scores": result.level_scores}


def _kill_stray_servers(ports: list[int]) -> None:
    for port in ports:
        subprocess.run(["fuser", "-k", f"{port}/tcp"], capture_output=True, check=False)


def run_arm(
    arm_key: str, call_log: list[dict[str, Any]], games: tuple[str, ...], seeds: tuple[int, ...]
) -> dict[str, Any]:
    cfg = ARMS[arm_key]
    _progress(f"arm {arm_key}: {cfg['label']}")
    os.environ["CARNOT_LLAMA_SERVER"] = str(cfg["server"])
    os.environ["CARNOT_ARC_LLAMA_SERVER_PARALLEL"] = "1"
    os.environ["CARNOT_ARC_INDUCE_MAX_TOKENS"] = str(INDUCE_MAX_TOKENS)
    os.environ["CARNOT_ARC_INDUCE_TIMEOUT"] = str(INDUCE_TIMEOUT_S)
    os.environ["CARNOT_ARC_INDUCE_N_CTX"] = str(INDUCE_N_CTX)
    proposer = _make_proposer(arm_key)
    baseline_cache: dict[str, list[int]] = {}
    rows: list[dict[str, Any]] = []
    for game in games:
        if game not in baseline_cache:
            baseline_cache[game] = _game_baseline_actions(game)
        for seed in seeds:
            _progress(f"  {arm_key} {game} seed={seed} starting")
            row = run_episode(proposer, game, seed, call_log)
            score = real_scorecard_score(baseline_cache[game], row)
            row.update(score)
            rows.append(row)
            _progress(
                f"  {arm_key} {game} seed={seed}: actions={row['actions_taken']} "
                f"levels={row['levels_completed']} score={row['game_score']:.4f} "
                f"llm_calls={row['llm_call_count']} tokens={row['llm_tokens_total']} "
                f"error={row['error']}"
            )
    try:
        proposer.stop()
    except Exception:  # noqa: BLE001 - best-effort cleanup
        pass
    _kill_stray_servers([cfg["port"]])
    time.sleep(2.0)
    total_llm_calls = sum(r["llm_call_count"] for r in rows)
    total_llm_tokens = sum(r["llm_tokens_total"] for r in rows)
    total_llm_duration = sum(r["llm_generation_duration_s"] for r in rows)
    return {
        "arm": arm_key,
        "label": cfg["label"],
        "server_binary": str(cfg["server"]),
        "gguf_path": str(cfg["gguf"]),
        "model_repository": cfg["repo"],
        "episodes": rows,
        "mean_game_score": statistics.mean(r["game_score"] for r in rows) if rows else 0.0,
        "per_game_mean_score": {
            game: statistics.mean(r["game_score"] for r in rows if r["game"] == game)
            for game in games
        },
        "any_level_up_games": sorted({r["game"] for r in rows if r["levels_completed"] > 0}),
        "total_llm_calls": total_llm_calls,
        "total_llm_tokens": total_llm_tokens,
        "total_llm_generation_duration_s": round(total_llm_duration, 3),
        "observed_tokens_per_second": (
            round(total_llm_tokens / total_llm_duration, 2) if total_llm_duration > 0 else None
        ),
        "episode_errors": [
            r["game"] + ":" + str(r["seed"]) + ":" + str(r["error"]) for r in rows if r["error"]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", default=list(ARMS.keys()), choices=list(ARMS.keys()))
    parser.add_argument("--games", nargs="+", default=list(GAMES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--out", default=str(RESULT_PATH))
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    _progress(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    _kill_stray_servers([cfg["port"] for cfg in ARMS.values()])
    # BUG FOUND 2026-09-29, mid-run: this used to read
    #   call_log: list[dict[str, Any]] = []
    #   _, restore = _install_instrumentation()
    # which throws away the real list `_install_instrumentation()` builds and
    # keeps a separate, permanently-empty one instead. Every real generate()/
    # complete_text() call landed in the discarded list. Real GPU time was
    # spent (one episode cost 307s) but every summary showed llm_calls=0 --
    # caught by comparing wall-clock cost against the recorded call count,
    # not by the code looking wrong on read-through.
    call_log, restore = _install_instrumentation()
    arm_results: dict[str, Any] = {}
    try:
        for arm_key in args.arms:
            arm_results[arm_key] = run_arm(arm_key, call_log, tuple(args.games), tuple(args.seeds))
            (RAW_DIR / f"{arm_key}.json").write_text(
                json.dumps(arm_results[arm_key], indent=2, sort_keys=True)
            )
    finally:
        restore()

    duration_s = round(time.monotonic() - started, 3)
    result = {
        "honest_verdict": "complete_arc_candidate_generator_eval_bonsai2_vs_mandated",
        "inference_substrate": "live_llm_inference",
        "inference_substrate_class": "model_full_generation",
        "solve_provenance": "development_proxy",
        "model_invoked": True,
        "live_model_loaded": True,
        "llm_call_count": sum(a["total_llm_calls"] for a in arm_results.values()),
        "generation_duration_s": round(
            sum(a["total_llm_generation_duration_s"] for a in arm_results.values()), 3
        ),
        "generation_invoked": True,
        "random_seeds_used": list(args.seeds),
        "games": list(args.games),
        "action_budget": ACTION_BUDGET,
        "episode_wall_cap_s": EPISODE_WALL_CAP_S,
        "induce_max_tokens": INDUCE_MAX_TOKENS,
        "induce_timeout_s": INDUCE_TIMEOUT_S,
        "induce_n_ctx": INDUCE_N_CTX,
        "preconditions_checked": {
            "gpu1_pinned": os.environ.get("CUDA_VISIBLE_DEVICES") == "1",
            "stock_llama_server_exists": STOCK_LLAMA_SERVER.exists(),
            "fork_llama_server_exists": FORK_LLAMA_SERVER.exists(),
            "gguf_standard_exists": GGUF_STANDARD.exists(),
            "gguf_ternary_exists": GGUF_TERNARY.exists(),
            "environment_files_dir": str(ENV_DIR),
        },
        "model_specs": {
            arm_key: {
                "model_repository": cfg["repo"],
                "model_filename": cfg["gguf"].name,
                "server_binary": str(cfg["server"]),
            }
            for arm_key, cfg in ARMS.items()
            if arm_key in arm_results
        },
        "arms": arm_results,
        "duration_s": duration_s,
    }
    result["reproducibility_checksum"] = _checksum(
        {k: v for k, v in result.items() if k != "reproducibility_checksum"}
    )
    out_path = Path(args.out)
    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    _progress(f"wrote {out_path} in {duration_s}s")


if __name__ == "__main__":
    main()

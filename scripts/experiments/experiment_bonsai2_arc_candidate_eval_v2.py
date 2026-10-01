"""Bonsai-2 ternary vs. mandated Qwen3.8-27B, ARC candidate-generator eval, version 2.

Why a second version (plain language).
    Version 1 (results/experiment_bonsai2_arc_candidate_eval.json) set the
    induction token cap to 512. Every induction call in both arms ran out of
    tokens and failed, so both arms played the same fallback explorer and the
    scores were identical. Nothing about model quality was measured.

What the diagnosis found (full evidence in the research note addendum).
    1. The live agent asks the server to think by default (think mode on, chat
       endpoint). The model then spends its whole budget inside hidden
       `reasoning_content`, and the answer channel stays empty. At 512 tokens
       there was never a chance of a valid answer.
    2. The live agent's own default budget is 131072 tokens. Its documented
       thinking inductions take 36k to 83k tokens. At about 40 tokens per
       second on one RTX 3090 that is 15 to 35 minutes per attempt. The full
       live-default setting does not fit an offline eval of this size.
    3. Even with thinking off, a valid engine needs about 2.5k to 3.5k
       tokens. So 512 was too small for both reasons.

What this version changes.
    It uses the live agent's own supported think-off switch
    (`CARNOT_ARC_INDUCE_THINK=0`, read by `induce_think_on()`). That switch
    selects the raw code-only completion path. No live-agent code or default
    is changed. The token cap, request timeout, context size and episode wall
    cap are raised so a valid induction can finish. They are identical in
    every arm.

What this version does NOT test.
    The live-default think-on mode at its full 131072 cap. That is a different,
    much slower question. Results here say nothing about it.

Everything else (games, seeds, disabled adapters, real scorecard formula,
live entrypoint E3AgentPolicy) is reused from version 1 by import.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # GPU 1 only, before any import that could touch CUDA.

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "python"))

import scripts.experiments.experiment_bonsai2_arc_candidate_eval as v1  # noqa: E402

RESULT_PATH = REPO_ROOT / "results/experiment_bonsai2_arc_candidate_eval_v2.json"
RAW_DIR = REPO_ROOT / "results/raw/experiment_bonsai2_arc_candidate_eval_v2"

# One budget for every arm. See the module docstring for the evidence behind each number.
INDUCE_MAX_TOKENS = 8192  # valid engines measured at 2.5k-3.5k tokens; 8192 leaves headroom
INDUCE_TIMEOUT_S = 600  # per request; 8192 tokens at 40 tok/s is about 205 s plus prefill
INDUCE_N_CTX = 32768  # one slot owns the pool; prompt (up to ~9k) + 8192 fits
EPISODE_WALL_CAP_S = 1200
ACTION_BUDGET = 300
THINK_ENV_VALUE = "0"  # the live agent's own think-off switch

GAMES = ("cd82", "lp85", "ar25")
SEEDS = (7491001, 7491002)
ARM_ORDER = ("A_stock_standard", "B_fork_standard", "C_fork_ternary")


def _progress(msg: str) -> None:
    print(f"[bonsai2-arc-v2 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _apply_config() -> None:
    """Point the v1 harness at the v2 settings. v1 functions read these module globals."""

    v1.INDUCE_MAX_TOKENS = INDUCE_MAX_TOKENS
    v1.INDUCE_TIMEOUT_S = INDUCE_TIMEOUT_S
    v1.INDUCE_N_CTX = INDUCE_N_CTX
    v1.RAW_DIR = RAW_DIR  # the engine store and attempt evidence land under the v2 raw folder
    os.environ["CARNOT_ARC_INDUCE_THINK"] = THINK_ENV_VALUE
    os.environ["CARNOT_ARC_LLAMA_SERVER_PARALLEL"] = "1"
    os.environ["CARNOT_ARC_INDUCE_MAX_TOKENS"] = str(INDUCE_MAX_TOKENS)
    os.environ["CARNOT_ARC_INDUCE_TIMEOUT"] = str(INDUCE_TIMEOUT_S)
    os.environ["CARNOT_ARC_INDUCE_N_CTX"] = str(INDUCE_N_CTX)


def _episode_path(arm_key: str, game: str, seed: int) -> Path:
    return RAW_DIR / "episodes" / f"{arm_key}__{game}__seed-{seed}.json"


_GATE_KEYS = (
    "planned",
    "skipped",
    "trust_metric",
    "verify_accuracy",
    "verify_cell_recall",
    "heldout_accuracy",
    "binary_gate_pass",
)


def _install_policy_capture():
    """Keep each E3AgentPolicy instance and count which phase every action came from.

    An induction call that returns code (`ok=True`) is not the same as an induced world
    model the agent trusts and plans with. The policy records that second fact in
    `induction_attempts`. This capture reads it without changing any live behavior.
    """

    from carnot.agentic import arc_competition_agent as live

    captured: list[Any] = []
    original = live.E3AgentPolicy

    class CapturingPolicy(original):  # type: ignore[misc, valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._eval_phase_counts: dict[str, int] = {}
            captured.append(self)

        def next_move(self, frames: Any, latest: Any) -> Any:
            result = super().next_move(frames, latest)
            phase = str(getattr(self, "phase", "unknown"))
            self._eval_phase_counts[phase] = self._eval_phase_counts.get(phase, 0) + 1
            return result

    live.E3AgentPolicy = CapturingPolicy

    def restore() -> None:
        live.E3AgentPolicy = original

    return captured, restore


def _policy_telemetry(policy: Any) -> dict[str, Any]:
    attempts = list(getattr(policy, "induction_attempts", []) or [])
    summaries = []
    for a in attempts:
        row = {k: a.get(k) for k in _GATE_KEYS if k in a}
        note = a.get("proposer_note")
        if note:
            row["proposer_note"] = str(note)[:400]
        summaries.append(row)
    try:
        witness_llm = dict(policy.generator_liveness_witness().get("llm") or {})
    except Exception as exc:  # noqa: BLE001 - telemetry must never break an episode
        witness_llm = {"error": repr(exc)[:200]}
    return {
        "induction_attempts_n": len(attempts),
        "induction_attempts_planned": sum(1 for a in attempts if a.get("planned")),
        "induction_attempt_summaries": summaries,
        "phase_action_counts": dict(getattr(policy, "_eval_phase_counts", {})),
        "generator_witness_llm": witness_llm,
    }


def run_arm(
    arm_key: str,
    call_log: list[dict[str, Any]],
    games: tuple[str, ...],
    seeds: tuple[int, ...],
    captured: list[Any],
) -> None:
    """Run every missing episode for one arm. Each episode is written to disk at once."""

    cfg = v1.ARMS[arm_key]
    # One engine store per arm. A shared store would let one arm's saved engine sit where the
    # next arm reads. Each arm's evidence also stays attributable to that arm.
    v1.RAW_DIR = RAW_DIR / "arms" / arm_key
    todo = [(g, s) for g in games for s in seeds if not _episode_path(arm_key, g, s).exists()]
    if not todo:
        _progress(f"arm {arm_key}: all episodes already on disk")
        return
    _progress(f"arm {arm_key}: {cfg['label']} ({len(todo)} episodes to run)")
    os.environ["CARNOT_LLAMA_SERVER"] = str(cfg["server"])
    proposer = v1._make_proposer(arm_key)
    (RAW_DIR / "episodes").mkdir(parents=True, exist_ok=True)
    baseline_cache: dict[str, list[int]] = {}
    for game, seed in todo:
        if game not in baseline_cache:
            baseline_cache[game] = v1._game_baseline_actions(game)
        _progress(f"  {arm_key} {game} seed={seed} starting")
        row = v1.run_episode(
            proposer,
            game,
            seed,
            call_log,
            budget=ACTION_BUDGET,
            wall_cap_s=EPISODE_WALL_CAP_S,
        )
        row.update(v1.real_scorecard_score(baseline_cache[game], row))
        row["arm"] = arm_key
        if captured:
            row.update(_policy_telemetry(captured[-1]))
        ok_calls = sum(1 for c in row["llm_calls"] if c["ok"])
        _progress(
            f"  {arm_key} {game} seed={seed}: actions={row['actions_taken']} "
            f"levels={row['levels_completed']} score={row['game_score']:.4f} "
            f"calls={row['llm_call_count']} ok={ok_calls} "
            f"planned={row.get('induction_attempts_planned')} tokens={row['llm_tokens_total']} "
            f"wall={row['duration_s']}s error={row['error']}"
        )
        _episode_path(arm_key, game, seed).write_text(
            json.dumps(row, indent=2, sort_keys=True, default=str)
        )
    try:
        proposer.stop()
    except Exception:  # noqa: BLE001 - best-effort cleanup
        pass
    v1._kill_stray_servers([cfg["port"]])
    time.sleep(2.0)


def _arm_summary(arm_key: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    calls = [c for r in rows for c in r["llm_calls"]]
    ok = [c for c in calls if c["ok"]]
    gen_s = sum(c["duration_s"] for c in calls)
    tokens = sum(c["tokens"] for c in calls if c["tokens"] > 0)
    games = sorted({r["game"] for r in rows})
    phase_totals: dict[str, int] = {}
    skip_reasons: dict[str, int] = {}
    failure_notes: dict[str, int] = {}
    for r in rows:
        for phase, n in (r.get("phase_action_counts") or {}).items():
            phase_totals[phase] = phase_totals.get(phase, 0) + int(n)
        for s in r.get("induction_attempt_summaries") or []:
            if s.get("skipped"):
                skip_reasons[str(s["skipped"])] = skip_reasons.get(str(s["skipped"]), 0) + 1
            if s.get("proposer_note"):
                key = str(s["proposer_note"])[:90]
                failure_notes[key] = failure_notes.get(key, 0) + 1
    return {
        "policy_induction_attempts_total": sum(r.get("induction_attempts_n", 0) for r in rows),
        "policy_induction_attempts_planned_total": sum(
            r.get("induction_attempts_planned", 0) for r in rows
        ),
        "policy_skip_reason_counts": skip_reasons,
        "policy_failure_note_counts": failure_notes,
        "policy_phase_action_totals": phase_totals,
        "arm": arm_key,
        "label": v1.ARMS[arm_key]["label"],
        "server_binary": str(v1.ARMS[arm_key]["server"]),
        "gguf_path": str(v1.ARMS[arm_key]["gguf"]),
        "model_repository": v1.ARMS[arm_key]["repo"],
        "n_episodes": len(rows),
        "mean_game_score": statistics.mean(r["game_score"] for r in rows) if rows else 0.0,
        "per_game_mean_score": {
            g: statistics.mean(r["game_score"] for r in rows if r["game"] == g) for g in games
        },
        "total_levels_completed": sum(r["levels_completed"] for r in rows),
        "level_up_games": sorted({r["game"] for r in rows if r["levels_completed"] > 0}),
        "mean_actions_taken": statistics.mean(r["actions_taken"] for r in rows) if rows else 0.0,
        "induction_call_count": len(calls),
        "induction_ok_count": len(ok),
        "induction_success_rate": (len(ok) / len(calls)) if calls else None,
        "total_generation_duration_s": round(gen_s, 3),
        # Effective rate: tokens of the LAST attempt of each call over the whole call time, which
        # includes prompt processing and any earlier failed attempts. It understates raw decode
        # speed (the server log shows about 40 tok/s for this model on one RTX 3090).
        "effective_tokens_per_second_during_play": round(tokens / gen_s, 2) if gen_s > 0 else None,
        "tokens_last_attempt_total": tokens,
        "mean_episode_wall_s": statistics.mean(r["duration_s"] for r in rows) if rows else 0.0,
        "episode_errors": [f"{r['game']}:{r['seed']}:{r['error']}" for r in rows if r["error"]],
    }


def _server_decode_tps(arm_key: str) -> dict[str, Any] | None:
    """Median decode speed the server itself reported during play (tokens per second).

    The server prints a `tg = N t/s` line every ~100 decoded tokens. These speeds are the
    clean per-token comparison. The client-side effective rate above mixes in prompt
    processing, failed attempts and idle time.
    """

    path = RAW_DIR / "server_logs" / f"{arm_key}.log"
    if not path.exists():
        return None
    speeds = [float(m) for m in re.findall(r"tg =\s+([0-9.]+) t/s", path.read_text())]
    if not speeds:
        return None
    return {
        "median_tokens_per_second": statistics.median(speeds),
        "min": min(speeds),
        "max": max(speeds),
        "n_samples": len(speeds),
        "source": str(path.relative_to(REPO_ROOT)),
    }


def _paired(rows_by_arm: dict[str, list[dict[str, Any]]], a: str, b: str) -> dict[str, Any]:
    """Paired difference (a minus b), matched on game and seed."""

    left = {(r["game"], r["seed"]): r for r in rows_by_arm.get(a, [])}
    right = {(r["game"], r["seed"]): r for r in rows_by_arm.get(b, [])}
    keys = sorted(set(left) & set(right))
    per_episode = [
        {
            "game": g,
            "seed": s,
            "score_diff": round(left[(g, s)]["game_score"] - right[(g, s)]["game_score"], 6),
            "levels_diff": left[(g, s)]["levels_completed"] - right[(g, s)]["levels_completed"],
        }
        for g, s in keys
    ]
    games = sorted({g for g, _ in keys})
    per_game = {
        g: round(statistics.mean(e["score_diff"] for e in per_episode if e["game"] == g), 6)
        for g in games
    }
    diffs = [e["score_diff"] for e in per_episode]
    return {
        "comparison": f"{a} minus {b}",
        "n_pairs": len(per_episode),
        "mean_score_diff": round(statistics.mean(diffs), 6) if diffs else None,
        "per_game_mean_score_diff": per_game,
        "per_episode": per_episode,
    }


def _truncation_diagnosis() -> dict[str, Any]:
    """Summarize the three probes that explain why version 1 truncated at 512 tokens.

    All probes sent the same real induction prompt (cd82, seed 7491001) to the stock
    server running the mandated model. The raw replies are saved next to this summary.
    """

    from carnot.agentic import arc_executable_world_model as e3

    base = RAW_DIR / "truncation_diagnosis"

    def chat(name: str) -> dict[str, Any] | None:
        p = base / name
        if not p.exists():
            return None
        d = json.loads(p.read_text())
        choice = d["choices"][0]
        msg = choice["message"]
        return {
            "finish_reason": choice.get("finish_reason"),
            "completion_tokens": d.get("usage", {}).get("completion_tokens"),
            "reasoning_content_chars": len(msg.get("reasoning_content") or ""),
            "content_chars": len(msg.get("content") or ""),
        }

    def raw(name: str) -> dict[str, Any] | None:
        p = base / name
        if not p.exists():
            return None
        d = json.loads(p.read_text())
        timings = d.get("timings", {})
        return {
            "stop_type": d.get("stop_type"),
            "predicted_n": timings.get("predicted_n"),
            "decode_tokens_per_second": timings.get("predicted_per_second"),
            "content_chars": len(d.get("content") or ""),
        }

    return {
        "finding": (
            "Both. The live agent asks for thinking by default, so a 512-token cap is spent "
            "entirely in hidden reasoning and the answer stays empty (hypothesis b). Even with "
            "thinking off, a valid engine needs 2.5k-3.5k tokens, so 512 was also too small "
            "(hypothesis a)."
        ),
        "probe_prompt": "cd82 seed 7491001 induction prompt, stock server, mandated model",
        "D1_chat_endpoint_default_think_cap512": chat("diag_A_D1_chat_default_512.json"),
        "D2_chat_endpoint_enable_thinking_false_cap4096": chat("diag_A_D2_chat_nothink_4096.json"),
        "D3_raw_codeonly_live_think_off_path_cap4096_mandated": raw(
            "diag_A_D3_raw_codeonly_4096.json"
        ),
        "D3_raw_codeonly_live_think_off_path_cap4096_ternary": raw(
            "diag_C_D3_raw_codeonly_4096.json"
        ),
        "ternary_model_chat_endpoint_probes": {
            "D1_chat_default_think_cap512": chat("diag_C_D1_chat_default_512.json"),
            "D2_chat_enable_thinking_false_cap4096": chat("diag_C_D2_chat_nothink_4096.json"),
            "finding": (
                "The ternary model also thinks by default (D1: answer channel empty at 512). "
                "On the chat endpoint with thinking off it writes a long, full engine (D2, hit "
                "the 4096 cap). On the raw code-only path that the think-off switch selects, it "
                "answered tersely and often omitted the engine function. So the low induction "
                "success of the ternary arm in this eval is partly a prompt-format effect of the "
                "raw path, not proof the weights cannot write the code. The chat path was probed "
                "once, not run through the 18-episode matrix."
            ),
        },
        "live_agent_defaults": {
            "think_mode_default_on": e3.ARC_LIVE_GENERATOR_THINK_SCORED_DEFAULT != "0",
            "induce_max_tokens_default": e3.ARC_LIVE_GENERATOR_INDUCE_MAX_TOKENS_DEFAULT,
            "induce_timeout_floor_s": e3.ARC_LIVE_GENERATOR_INDUCE_TIMEOUT_FLOOR_S,
            "documented_think_completion_tokens": (
                "36406-83444, median 62490 (comment above "
                "ARC_LIVE_GENERATOR_INDUCE_MAX_TOKENS_DEFAULT in arc_executable_world_model.py)"
            ),
            "think_flag_ignored_by_prior_harness": (
                "v1 passed no_think_prefix='/no_think', but generate() applies that prefix only "
                "when think mode is off. Think mode is on by default, so the prefix was inert."
            ),
        },
        "stock_server_note": (
            "chat_template_kwargs enable_thinking=false works on the stock server (D2). "
            "The live agent's own switch CARNOT_ARC_INDUCE_THINK=0 was used instead, so no "
            "live code changed."
        ),
    }


def aggregate(args: argparse.Namespace) -> None:
    rows_by_arm: dict[str, list[dict[str, Any]]] = {}
    for arm in ARM_ORDER:
        rows = []
        for game in args.games:
            for seed in args.seeds:
                p = _episode_path(arm, game, seed)
                if p.exists():
                    rows.append(json.loads(p.read_text()))
        if rows:
            rows_by_arm[arm] = rows
    arm_results = {arm: _arm_summary(arm, rows) for arm, rows in rows_by_arm.items()}
    for arm, rows in rows_by_arm.items():
        arm_results[arm]["episodes"] = rows
        arm_results[arm]["server_decode_speed"] = _server_decode_tps(arm)
    run1_dir = RAW_DIR / "run1_shared_engine_store_superseded"
    run1_sessions = (
        json.loads((run1_dir / "sessions.json").read_text())
        if (run1_dir / "sessions.json").exists()
        else []
    )
    sessions_path = RAW_DIR / "sessions.json"
    sessions = json.loads(sessions_path.read_text()) if sessions_path.exists() else []
    duration_s = round(sum(s["wall_s"] for s in sessions), 3)
    pairs = {}
    names = [a for a in ARM_ORDER if a in rows_by_arm]
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            pairs[f"{a}__vs__{b}"] = _paired(rows_by_arm, a, b)
    all_calls = [c for rows in rows_by_arm.values() for r in rows for c in r["llm_calls"]]
    result = {
        "honest_verdict": args.verdict,
        "inference_substrate": "live_llm_inference",
        "inference_substrate_class": "model_full_generation",
        "solve_provenance": "development_proxy",
        "model_invoked": True,
        "live_model_loaded": True,
        "generation_invoked": True,
        "llm_call_count": len(all_calls),
        "generation_duration_s": round(sum(c["duration_s"] for c in all_calls), 3),
        "random_seeds_used": list(args.seeds),
        "games": list(args.games),
        "action_budget": ACTION_BUDGET,
        "episode_wall_cap_s": EPISODE_WALL_CAP_S,
        "induce_max_tokens": INDUCE_MAX_TOKENS,
        "induce_timeout_s": INDUCE_TIMEOUT_S,
        "induce_n_ctx": INDUCE_N_CTX,
        "live_agent_think_mode_env": {"CARNOT_ARC_INDUCE_THINK": THINK_ENV_VALUE},
        "supersedes": "results/experiment_bonsai2_arc_candidate_eval.json",
        "truncation_diagnosis": _truncation_diagnosis(),
        "superseded_first_pass_of_this_version": {
            "note": (
                "A first full pass used one engine store shared by all three arms and no policy "
                "telemetry. It was discarded and rerun with one store per arm. Episode scores, "
                "call counts and token counts matched the rerun exactly for all 18 episodes "
                "(the runs are seeded), so the shared store did not change any result."
            ),
            "sessions": run1_sessions,
            "episodes_dir": "results/raw/experiment_bonsai2_arc_candidate_eval_v2/"
            "run1_shared_engine_store_superseded/episodes",
        },
        "preconditions_checked": {
            "gpu1_pinned": os.environ.get("CUDA_VISIBLE_DEVICES") == "1",
            "stock_llama_server_exists": v1.STOCK_LLAMA_SERVER.exists(),
            "fork_llama_server_exists": v1.FORK_LLAMA_SERVER.exists(),
            "gguf_standard_exists": v1.GGUF_STANDARD.exists(),
            "gguf_ternary_exists": v1.GGUF_TERNARY.exists(),
            "environment_files_dir": str(v1.ENV_DIR),
        },
        "model_specs": {
            arm: {
                "model_repository": v1.ARMS[arm]["repo"],
                "model_filename": v1.ARMS[arm]["gguf"].name,
                "server_binary": str(v1.ARMS[arm]["server"]),
            }
            for arm in rows_by_arm
        },
        "arms": arm_results,
        "paired_differences": pairs,
        "sessions": sessions,
        "duration_s": duration_s,
        "sample_size_note": (
            "3 games x 2 seeds x 3 arms. Directional only, not statistically powered. "
            "Each arm's two seeds per game are not independent draws of the game."
        ),
    }
    result["reproducibility_checksum"] = v1._checksum(
        {k: v for k, v in result.items() if k != "reproducibility_checksum"}
    )
    out = Path(args.out)
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    _progress(f"wrote {out} (duration_s={duration_s})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", default=list(ARM_ORDER), choices=list(ARM_ORDER))
    parser.add_argument("--games", nargs="+", default=list(GAMES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--out", default=str(RESULT_PATH))
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument(
        "--verdict",
        default=(
            "complete_arc_candidate_eval_v2_induction_completed_in_all_arms_"
            "ternary_induction_success_rate_low_scores_equal_directional_only"
        ),
    )
    args = parser.parse_args()

    _apply_config()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if args.aggregate_only:
        aggregate(args)
        return
    started = time.monotonic()
    _progress(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    v1._kill_stray_servers([cfg["port"] for cfg in v1.ARMS.values()])
    call_log, restore = v1._install_instrumentation()
    captured, restore_policy = _install_policy_capture()
    try:
        for arm in args.arms:
            run_arm(arm, call_log, tuple(args.games), tuple(args.seeds), captured)
    finally:
        restore_policy()
        restore()
    wall_s = round(time.monotonic() - started, 3)
    sessions_path = RAW_DIR / "sessions.json"
    sessions = json.loads(sessions_path.read_text()) if sessions_path.exists() else []
    sessions.append(
        {
            "arms": list(args.arms),
            "games": list(args.games),
            "seeds": list(args.seeds),
            "wall_s": wall_s,
            "ended": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    )
    sessions_path.write_text(json.dumps(sessions, indent=2))
    _progress(f"session done in {wall_s}s")


if __name__ == "__main__":
    main()

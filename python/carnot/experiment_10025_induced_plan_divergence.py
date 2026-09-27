"""REQ-ARC-WMTE-10025: bounded real-GPU live induced-plan pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PARENT = Path("/home/ianblenke/github.com/ianblenke/carnot")
RESULT = ROOT / "results/experiment_10025_induced_plan_divergence.json"
RAW = ROOT / "results/raw/experiment_10025_induced_plan_divergence"
MODEL_ID = "unsloth/Qwen3.8-27B-GGUF"
MODEL_FILE = "Qwen3.8-27B-Q4_K_M.gguf"
MODEL_CACHE = Path.home() / ".cache/huggingface/hub/models--unsloth--Qwen3.8-27B-GGUF"
CUDA_SERVER = Path.home() / ".cache/llama.cpp-master/build/bin/llama-server"
GAMES = ("su15", "sp80", "g50t", "cd82")
SEEDS = (7491001, 7491002)
ARMS = (
    ("1.0_no_halt", 1.0, False),
    ("1.0_halt", 1.0, True),
    ("0.9_halt", 0.9, True),
    ("0.75_halt", 0.75, True),
)
ACTION_LIMIT = 400
RANDOM_SEED = 10025


def progress(message: str) -> None:
    """REQ-ARC-WMTE-10025 prints a flushed live progress receipt."""

    print(f"[exp10025 {time.strftime('%H:%M:%S')}] {message}", flush=True)


def _checksum(document: dict[str, Any]) -> str:
    payload = {key: value for key, value in document.items() if key != "reproducibility_checksum"}
    return (
        "sha256:"
        + hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()
    )


def _write(document: dict[str, Any]) -> None:
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    document["reproducibility_checksum"] = _checksum(document)
    with tempfile.NamedTemporaryFile(mode="w", dir=RESULT.parent, delete=False) as stream:
        json.dump(document, stream, indent=2, sort_keys=True, default=str)
        temp = Path(stream.name)
    temp.replace(RESULT)


def _preconditions() -> tuple[list[dict[str, Any]], Path | None]:
    """SCENARIO-ARC-WMTE-10025-THRESHOLDS admits only cached GPU-1 inference."""

    checks: list[dict[str, Any]] = []
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    checks.append({"name": "physical_gpu_1_only", "passed": visible == "1", "observed": visible})
    snapshots = list((MODEL_CACHE / "snapshots").glob(f"*/{MODEL_FILE}"))
    model = snapshots[0].resolve() if snapshots else None
    checks.append(
        {
            "name": "model_cached",
            "passed": bool(model and model.is_file()),
            "observed": str(model) if model else None,
        }
    )
    gpu = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    lines = [line.strip() for line in gpu.stdout.splitlines()]
    used = None
    for line in lines:
        fields = [part.strip() for part in line.split(",")]
        if len(fields) >= 3 and fields[0] == "1":
            used = int(fields[1])
    checks.append(
        {
            "name": "gpu_1_idle",
            "passed": gpu.returncode == 0 and used is not None and used < 500,
            "observed_mb": used,
            "raw": lines,
        }
    )
    try:
        import llama_cpp

        offload = bool(llama_cpp.llama_supports_gpu_offload())
    except Exception as exc:
        offload = False
        checks.append({"name": "llama_cpp_import", "passed": False, "error": repr(exc)[:200]})
    checks.append({"name": "llama_cpp_gpu_offload", "passed": offload})
    checks.append(
        {
            "name": "cuda_llama_server_binary",
            "passed": CUDA_SERVER.is_file() and os.access(CUDA_SERVER, os.X_OK),
            "path": str(CUDA_SERVER),
        }
    )
    env_dir = PARENT / "environment_files"
    checks.append(
        {"name": "parent_environment_files", "passed": env_dir.is_dir(), "path": str(env_dir)}
    )
    return checks, model


def _owned_gpu_memory_mb(pid: int | None) -> int:
    """REQ-ARC-WMTE-10025 refuses a healthy but CPU-only llama-server."""

    if pid is None:
        return 0
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    for line in result.stdout.splitlines():
        fields = [part.strip() for part in line.split(",")]
        if len(fields) >= 2 and fields[0] == str(pid):
            return int(fields[1])
    return 0


def _base_artifact(checks: list[dict[str, Any]], model: Path | None) -> dict[str, Any]:
    return {
        "schema": "carnot.experiment_10025_induced_plan_divergence.v1",
        "requirement_id": "REQ-ARC-WMTE-10025",
        "honest_verdict": "incomplete_measurement",
        "inference_substrate": "live_llm_inference",
        "solve_provenance": "live_agent_self_discovery",
        "random_seed": RANDOM_SEED,
        "seeds": list(SEEDS),
        "games": list(GAMES),
        "arms": [
            {"name": name, "threshold": threshold, "halt": halt} for name, threshold, halt in ARMS
        ],
        "model_specs": {
            "repository": MODEL_ID,
            "filename": MODEL_FILE,
            "path": str(model) if model else None,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "n_ctx": 98_304,
            "server_parallel": 1,
        },
        "preconditions_checked": checks,
        "action_limit": ACTION_LIMIT,
        "episode_bounds": "first_level_up_or_400_actions_before_second_induction",
        "duration_s": 0.0,
        "episodes": [],
        "promotion": {},
        "reproducibility_checksum": "",
    }


def _score(arcade: Any, scorecard_id: str, game: str) -> tuple[float, dict[str, Any]]:
    """REQ-ARC-WMTE-10025 reads the installed scorecard's per-game formula output."""

    card = arcade.get_scorecard(scorecard_id)
    if card is None:
        raise RuntimeError("installed scorecard returned None")
    matches = [env for env in card.environments if str(env.id).split("-")[0] == game]
    if not matches:
        return 0.0, {"reason": "no_scored_run", "scorecard_id": scorecard_id}
    selected = matches[0]
    return float(selected.score), {
        "scorecard_id": scorecard_id,
        "runs": [run.model_dump(mode="json") for run in selected.runs],
    }


def _plan_acceptance_path(attempts: list[dict[str, Any]]) -> str:
    """REQ-ARC-WMTE-10025 separates bounded acceptance from later fallback plans."""

    if any(row.get("planned") and not row.get("skipped") for row in attempts):
        return "bounded_verified_plan"
    if any(row.get("planned") and row.get("skipped") for row in attempts):
        return "post_rejection_plan"
    return "no_plan"


def _episode(game: str, seed: int, arm: tuple[str, float, bool], proposer: Any) -> dict[str, Any]:
    """SCENARIO-ARC-WMTE-10025-DIVERGENCE runs the scored E3 agent on a real simulator."""

    from carnot.agentic import arc_executable_world_model as e3
    from carnot.agentic import arc_solver_kit as kit
    from carnot.agentic.arc_competition_agent import make_carnot_agent
    from carnot import experiment_7471_v654_arc_seam_observation as exp7471

    name, threshold, halt = arm
    episode_id = f"{game}__{seed}__{name}"
    directory = RAW / episode_id
    directory.mkdir(parents=True, exist_ok=True)
    os.environ["CARNOT_ARC_INDUCTION_ACCEPT_THRESHOLD"] = str(threshold)
    os.environ["CARNOT_ARC_ACTION_PROVENANCE"] = "1"
    os.environ["CARNOT_ARC_RANDOM_SEED"] = str(seed)
    os.environ["CARNOT_ARC_GENERATOR_SEED"] = str(seed)
    if halt:
        os.environ["CARNOT_ARC_PLAN_DIVERGENCE_HALT"] = "1"
    else:
        os.environ.pop("CARNOT_ARC_PLAN_DIVERGENCE_HALT", None)
    random.seed(seed)
    import numpy as np

    np.random.seed(seed % (2**32 - 1))
    old_store = e3.E3_DIR
    e3.E3_DIR = directory / "fresh_engine_store"
    kit.ENV_DIR = PARENT / "environment_files"

    class LocalAgentBase:
        def __init__(self, game_id: str) -> None:
            self.game_id = game_id

    began = time.monotonic()
    phase = {"name": "setup", "action": 0, "induction_calls": 0}
    heartbeat_stop = threading.Event()

    def heartbeat() -> None:
        while not heartbeat_stop.wait(45):
            progress(
                f"heartbeat {episode_id} phase={phase['name']} action={phase['action']} induction_calls={phase['induction_calls']} elapsed_s={time.monotonic() - began:.1f}"
            )

    ticker = threading.Thread(target=heartbeat, daemon=True)
    ticker.start()
    action_rows: list[dict[str, Any]] = []
    error: str | None = None
    policy = None
    score = 0.0
    score_detail: dict[str, Any] = {}
    level_ups: list[dict[str, int]] = []
    stop_reason = "action_limit"
    try:
        originals = exp7471._disable_cross_game_loaders()
        try:
            agent_type = make_carnot_agent(LocalAgentBase, cascade=True, proposer=proposer)
            live_agent = agent_type(game_id=game)
        finally:
            exp7471._restore_cross_game_loaders(originals)
        policy = live_agent._policy

        def induction_hook(kind: str, payload: dict[str, Any]) -> None:
            phase["name"] = kind
            if kind == "induction_started":
                phase["induction_calls"] += 1
            progress(f"{episode_id} {kind} {json.dumps(payload, default=str)[:500]}")

        policy.induction_progress_hook = induction_hook
        arcade = kit.offline_arcade()
        scorecard_id = arcade.open_scorecard()
        env = arcade.make(game, scorecard_id=scorecard_id)
        frames: list[Any] = []
        latest: Any = None
        peak = 0
        phase["name"] = "actions"
        with (directory / "actions.jsonl").open("w", encoding="utf-8") as stream:
            for action_index in range(ACTION_LIMIT):
                if peak > 0:
                    stop_reason = "first_level_up"
                    break
                if (
                    policy._induction_attempt_count >= 1
                    and policy.phase == "induce"
                    and not policy.plan
                ):
                    stop_reason = "before_repeat_induction"
                    break
                if live_agent.is_done(frames, latest):
                    stop_reason = "agent_done"
                    break
                phase["action"] = action_index + 1
                action = live_agent.choose_action(frames, latest)
                action_name = str(getattr(action, "name", action))
                data_value = getattr(action, "action_data", None)
                data = data_value.model_dump() if hasattr(data_value, "model_dump") else None
                if isinstance(data, dict):
                    data = {key: value for key, value in data.items() if key != "game_id"}
                latest = env.reset() if action_name == "RESET" else env.step(action, data=data)
                level = int(getattr(latest, "levels_completed", 0) or 0)
                if level > peak:
                    for crossed in range(peak + 1, level + 1):
                        level_ups.append({"level": crossed, "action_index": action_index + 1})
                    peak = level
                row = {
                    "action_index": action_index + 1,
                    "action": action_name,
                    "data": data,
                    "level": level,
                    "phase": policy.phase,
                    "plan_pi": policy.pi,
                    "plan_len": len(policy.plan),
                }
                action_rows.append(row)
                stream.write(json.dumps(row, default=str) + "\n")
                stream.flush()
                frames.append(latest)
                if (action_index + 1) % 100 == 0:
                    progress(
                        f"{episode_id} actions={action_index + 1} level={peak} plans={len(policy.verified_divergence_events)}"
                    )
        score, score_detail = _score(arcade, scorecard_id, game)
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"[:500]
        progress(f"{episode_id} error={error}")
    finally:
        heartbeat_stop.set()
        ticker.join(timeout=1)
        e3.E3_DIR = old_store
    attempts = list(policy.induction_attempts) if policy is not None else []
    (directory / "induction_attempts.json").write_text(
        json.dumps(attempts, indent=2, default=str), encoding="utf-8"
    )
    provenance = policy.action_provenance() if policy is not None else None
    if provenance is not None:
        (directory / "action_provenance.json").write_text(
            json.dumps(provenance.to_dict(), indent=1, default=str), encoding="utf-8"
        )
    return {
        "episode_id": episode_id,
        "game": game,
        "seed": seed,
        "arm": name,
        "threshold": threshold,
        "divergence_halt_enabled": halt,
        "status": "complete" if error is None else "error",
        "error": error,
        "duration_s": round(time.monotonic() - began, 6),
        "real_scorecard_score": score,
        "scorecard": score_detail,
        "real_actions": len(action_rows),
        "termination_reason": stop_reason,
        "level_ups": level_ups,
        "plan_accepted": any(bool(row.get("planned")) for row in attempts),
        "plan_acceptance_path": _plan_acceptance_path(attempts),
        "bounded_plan_accepted": _plan_acceptance_path(attempts) == "bounded_verified_plan",
        "plan_executed": bool(provenance and any(row.get("plan_step") for row in provenance.rows)),
        "plan_actions_executed": (
            sum(bool(row.get("plan_step")) for row in provenance.rows) if provenance else 0
        ),
        "plan_completed": bool(provenance and provenance.plans_consumed_fully),
        "divergence_events": list(policy.verified_divergence_events) if policy is not None else [],
        "plans_abandoned_by_verified_divergence": (
            policy.plans_abandoned_by_verified_divergence if policy is not None else 0
        ),
        "induction_attempts": attempts,
        "induction_acceptance_rejection_reasons": [
            str(row.get("skipped") or "accepted") for row in attempts
        ],
        "provenance_summary": provenance.summary() if provenance is not None else None,
        "raw_actions": str((directory / "actions.jsonl").relative_to(ROOT)),
        "raw_provenance": str((directory / "action_provenance.json").relative_to(ROOT)),
    }


def _promotion(rows: list[dict[str, Any]], games: tuple[str, ...]) -> dict[str, Any]:
    """REQ-ARC-WMTE-10025 applies the fixed paired real-scorecard promotion rule."""

    indexed = {
        (row["game"], row["seed"], row["arm"]): row for row in rows if row["status"] == "complete"
    }
    result: dict[str, Any] = {}
    keys = [(game, seed) for game in games for seed in SEEDS]
    for name, _threshold, _halt in ARMS:
        if name == "1.0_no_halt":
            continue
        pairs = [(indexed.get((*key, "1.0_halt")), indexed.get((*key, name))) for key in keys]
        complete = all(left is not None and right is not None for left, right in pairs)
        if not complete:
            result[name] = {"eligible": False, "reason": "missing_paired_episode"}
            continue
        baseline = sum(float(left["real_scorecard_score"]) for left, _ in pairs) / len(pairs)
        treatment = sum(float(right["real_scorecard_score"]) for _, right in pairs) / len(pairs)
        drops = [
            f"{left['game']}:{left['seed']}"
            for left, right in pairs
            if float(right["real_scorecard_score"]) < float(left["real_scorecard_score"])
        ]
        result[name] = {
            "eligible": True,
            "control_mean": baseline,
            "arm_mean": treatment,
            "score_drops": drops,
            "promotion_candidate": name != "1.0_halt" and treatment > baseline and not drops,
        }
    return result


def run_experiment(games: tuple[str, ...] = GAMES, max_hours: float = 3.0) -> dict[str, Any]:
    """REQ-ARC-WMTE-10025 runs whole-game blocks under the real GPU preconditions."""

    started = time.monotonic()
    progress("phase=preconditions")
    checks, model = _preconditions()
    artifact = _base_artifact(checks, model)
    artifact["scheduled_games"] = list(games)
    artifact["scope_cut"] = [game for game in GAMES if game not in games]
    if not all(check.get("passed") for check in checks):
        failed = next(check for check in checks if not check.get("passed"))
        artifact["honest_verdict"] = f"blocked_{failed['name']}"
        artifact["duration_s"] = round(time.monotonic() - started, 6)
        _write(artifact)
        return artifact
    _write(artifact)
    assert model is not None
    os.environ["CARNOT_LLAMA_SERVER"] = str(CUDA_SERVER)
    os.environ["CARNOT_ARC_LLAMA_SERVER_PARALLEL"] = "1"
    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

    proposer = LocalGGUFProposer(
        repo_substr="Qwen3.8-27B",
        model_path=str(model),
        port=8925,
        mtp=False,
        kv_quant="q8_0",
        use_chat_template=True,
        n_gpu_layers=999,
        n_ctx=98_304,
        tries=1,
    )
    proposer.model_repository = MODEL_ID
    proposer.requested_model_filename = MODEL_FILE
    proposer.requested_model_path = str(model)
    generate = proposer.generate
    generation_calls = 0

    def logged_generate(*args: Any, **kwargs: Any) -> Any:
        nonlocal generation_calls
        generation_calls += 1
        progress(f"induction_generation_call={generation_calls} started")
        try:
            return generate(*args, **kwargs)
        finally:
            progress(f"induction_generation_call={generation_calls} finished")

    proposer.generate = logged_generate
    with tempfile.TemporaryDirectory(prefix="carnot_exp10025_") as scratch:
        old_cwd = Path.cwd()
        os.chdir(scratch)
        try:
            progress("phase=model_load")
            if not proposer._ensure_server():
                raise RuntimeError("native CUDA llama-server failed to start")
            artifact["model_specs"]["server_pid"] = getattr(proposer._proc, "pid", None)
            artifact["model_specs"]["server_command"] = list(proposer.last_launch_argv)
            owned_gpu_mb = _owned_gpu_memory_mb(artifact["model_specs"]["server_pid"])
            artifact["model_specs"]["owned_gpu_memory_mb"] = owned_gpu_mb
            artifact["preconditions_checked"].append(
                {
                    "name": "runtime_server_gpu_offload",
                    "passed": owned_gpu_mb >= 8000,
                    "server_pid": artifact["model_specs"]["server_pid"],
                    "owned_gpu_memory_mb": owned_gpu_mb,
                    "minimum_mb": 8000,
                }
            )
            if owned_gpu_mb < 8000:
                raise RuntimeError(f"runtime_server_gpu_offload insufficient: {owned_gpu_mb} MiB")
            progress(f"phase=model_loaded server_pid={artifact['model_specs']['server_pid']}")
            for game in games:
                if time.monotonic() - started >= max_hours * 3600:
                    artifact["scope_cut"].extend(
                        [
                            item
                            for item in games
                            if item not in {row["game"] for row in artifact["episodes"]}
                        ]
                    )
                    progress(f"phase=whole_game_wall_clock_cut next={game}")
                    break
                progress(f"phase=game_start game={game}")
                for seed in SEEDS:
                    for arm in ARMS:
                        progress(f"phase=episode_start game={game} seed={seed} arm={arm[0]}")
                        row = _episode(game, seed, arm, proposer)
                        artifact["episodes"].append(row)
                        artifact["duration_s"] = round(time.monotonic() - started, 6)
                        _write(artifact)
                        progress(
                            f"phase=episode_end id={row['episode_id']} score={row['real_scorecard_score']:.6f} actions={row['real_actions']} status={row['status']}"
                        )
                progress(f"phase=game_end game={game}")
        except Exception as exc:
            artifact["run_error"] = f"{type(exc).__name__}: {exc}"[:500]
            artifact["honest_verdict"] = (
                "blocked_runtime_gpu_offload"
                if "runtime_server_gpu_offload" in str(exc)
                else "blocked_model_load"
                if not artifact["episodes"]
                else "incomplete_run_error"
            )
        finally:
            proposer.stop()
            os.chdir(old_cwd)
    measured_games = tuple(
        game
        for game in games
        if all(
            any(
                row["game"] == game and row["seed"] == seed and row["arm"] == arm[0]
                for row in artifact["episodes"]
            )
            for seed in SEEDS
            for arm in ARMS
        )
    )
    artifact["promotion"] = (
        _promotion(artifact["episodes"], measured_games) if measured_games else {}
    )
    artifact["duration_s"] = round(time.monotonic() - started, 6)
    if artifact["honest_verdict"] == "incomplete_measurement":
        any_candidate = any(
            row.get("promotion_candidate") for row in artifact["promotion"].values()
        )
        artifact["honest_verdict"] = (
            "complete_promotion_candidate" if any_candidate else "complete_no_promotion"
        )
        if len(measured_games) < len(games):
            artifact["honest_verdict"] = "incomplete_whole_game_scope_cut"
    _write(artifact)
    progress(
        f"phase=terminal verdict={artifact['honest_verdict']} episodes={len(artifact['episodes'])}"
    )
    return artifact


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", nargs="+", choices=GAMES, default=list(GAMES))
    parser.add_argument("--max-hours", type=float, default=3.0)
    args = parser.parse_args()
    result = run_experiment(tuple(args.games), args.max_hours)
    return 0 if str(result["honest_verdict"]).startswith("complete_") else 1

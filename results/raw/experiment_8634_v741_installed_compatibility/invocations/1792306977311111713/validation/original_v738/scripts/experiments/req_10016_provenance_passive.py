#!/usr/bin/env python3
"""REQ-ARC-WMTE-10016: CPU offline replay and main-file decision differential."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import tempfile
import time
import types
from pathlib import Path
from typing import Any

import numpy as np

from carnot.agentic import arc_executable_world_model as e3
from carnot.agentic import arc_solver_kit as kit
from carnot.agentic import arc_competition_agent as current_agent
from carnot.agentic.arc_agi3_world_model import frame_hash, grid_of
from carnot.experiment_10012_gate_usefulness import resolve_environment_files

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "results/raw/req_10016_provenance_passive"
MAIN_FILE = "python/carnot/agentic/arc_competition_agent.py"
GAMES = ("vc33", "sp80", "ft09", "g50t", "tu93")
SEEDS = (7491001, 7491002)
ACTION_CAP = 200


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def _load_main_agent():
    source = subprocess.run(
        ["git", "show", f"main:{MAIN_FILE}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    module = types.ModuleType("_req10016_main_agent")
    module.__file__ = str(ROOT / MAIN_FILE)
    module.__package__ = "carnot.agentic"
    sys.modules[module.__name__] = module
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module, hashlib.sha256(source.encode()).hexdigest()


def _disable_cross_game_loaders(agent) -> None:
    agent._recommend_live_approach = lambda _game, **_kw: {
        "strategy": {"uses_goal_distance_heuristic": False},
        "source": "req10016_anonymous_runtime_only",
    }
    agent.load_cross_game_value_head = lambda: None
    agent._load_submitted_candidate_router = lambda game_id=None: None
    agent._load_submitted_goal_energy_bias = lambda: None


def episode(game: str, seed: int, arm: str, cap: int) -> dict[str, Any]:
    """Run the scored E3 policy against public offline frames with no stored routes."""
    from arcengine import GameAction

    os.environ["CARNOT_ARC_DISABLE_INDUCTION"] = "1"
    os.environ["CARNOT_ARC_RANDOM_SEED"] = str(seed)
    os.environ["CARNOT_ARC_GENERATOR_SEED"] = str(seed)
    if arm == "on":
        os.environ["CARNOT_ARC_ACTION_PROVENANCE"] = "1"
    else:
        os.environ.pop("CARNOT_ARC_ACTION_PROVENANCE", None)
    random.seed(seed)
    np.random.seed(seed)
    kit.ENV_DIR = resolve_environment_files(ROOT)
    if not kit.ENV_DIR.is_dir():
        raise FileNotFoundError(kit.ENV_DIR)
    agent = current_agent
    baseline_sha256 = None
    if arm == "main":
        agent, baseline_sha256 = _load_main_agent()
    _disable_cross_game_loaders(agent)
    from carnot.agentic import arc_game_adapters

    def forbidden_adapter(_game):
        raise AssertionError("per-game adapter entered passive provenance proof")

    arc_game_adapters.get_adapter = forbidden_adapter
    e3.E3_DIR = Path.cwd() / "empty_engine_store"
    policy = agent.E3AgentPolicy(
        "unseen_runtime_game", proposer=None, frontier_discipline_seed=seed
    )
    arcade = kit.offline_arcade()
    env = arcade.make(game, scorecard_id=arcade.open_scorecard())
    frames: list[Any] = []
    latest = None
    trace: list[dict[str, Any]] = []
    RAW.mkdir(parents=True, exist_ok=True)
    out = RAW / f"{game}__seed-{seed}__{arm}__cap-{cap}.jsonl"
    with out.open("w") as writer:
        for index in range(1, cap + 1):
            if policy.is_done(frames, latest):
                break
            kind, data = policy.next_move(frames, latest)
            if kind is None:
                break
            latest = (
                env.reset()
                if kind == "RESET"
                else env.step(getattr(GameAction, f"ACTION{kind}"), data=data)
            )
            if arm == "on":
                policy.observe_action_outcome(latest)
            frames.append(latest)
            action = {"action": kind, "data": data}
            trace.append(action)
            recorder = policy.action_provenance()
            provenance = recorder.rows[-1] if recorder is not None else {}
            row = {
                "action_index": index,
                **action,
                "levels_completed": int(getattr(latest, "levels_completed", 0) or 0),
                "grid_hash": frame_hash(grid_of(latest)),
                "phase": provenance.get("phase_before"),
                "top_branch": provenance.get("top_branch"),
                "explorer_branch": provenance.get("explorer_branch"),
                "explorer_serve_kind": provenance.get("explorer_serve_kind"),
                "plan_step": provenance.get("plan_step"),
                "recorded_levels_completed": provenance.get("levels_completed"),
            }
            if arm == "on" and row["recorded_levels_completed"] != row["levels_completed"]:
                raise AssertionError(f"resulting frame did not join action {index}")
            writer.write(canonical(row) + "\n")
    return {
        "game": game,
        "seed": seed,
        "arm": arm,
        "actions": len(trace),
        "trace_sha256": digest(trace),
        "raw_trace": str(out.relative_to(ROOT)),
        "main_file_sha256": baseline_sha256,
    }


def _run_worker(game: str, seed: int, arm: str, cap: int) -> dict[str, Any]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["JAX_PLATFORMS"] = "cpu"
    env["PYTHONPATH"] = str(ROOT / "python")
    with tempfile.TemporaryDirectory(prefix="req10016-") as scratch:
        env["CARNOT_ARC_ACTION_PROVENANCE_DIR"] = scratch
        env["MPLCONFIGDIR"] = scratch
        started = time.monotonic()
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--game",
                game,
                "--seed",
                str(seed),
                "--arm",
                arm,
                "--cap",
                str(cap),
            ],
            cwd=scratch,
            env=env,
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
        if result.returncode:
            raise RuntimeError(
                f"{game} {seed} {arm}: exit={result.returncode}\n"
                f"stdout={result.stdout[-2000:]}\nstderr={result.stderr[-2000:]}"
            )
        row = json.loads(result.stdout.splitlines()[-1])
        row["duration_s"] = round(time.monotonic() - started, 3)
        return row


def run_proof(*, differential_only: bool = False) -> dict[str, Any]:
    """SCENARIO-ARC-WMTE-10016-IDENTITY: compare full action/data streams."""
    RAW.mkdir(parents=True, exist_ok=True)
    if differential_only:
        identity = json.loads((RAW / "recording_identity.json").read_text())
        pairs = identity["pairs"]
        if len(pairs) != len(GAMES) * len(SEEDS) or not all(row["identical"] for row in pairs):
            raise AssertionError("saved identity proof incomplete")
    else:
        pairs = []
        for game in GAMES:
            for seed in SEEDS:
                off = _run_worker(game, seed, "off", ACTION_CAP)
                on = _run_worker(game, seed, "on", ACTION_CAP)
                same = off["actions"] == on["actions"] and off["trace_sha256"] == on["trace_sha256"]
                pair = {"game": game, "seed": seed, "off": off, "on": on, "identical": same}
                pairs.append(pair)
                print(f"identity {game} seed={seed}: {same}, actions={on['actions']}", flush=True)
                if not same:
                    raise AssertionError(f"recording changed {game} seed={seed}")
        identity = {
            "requirement": "REQ-ARC-WMTE-10016",
            "games": list(GAMES),
            "seeds": list(SEEDS),
            "action_cap": ACTION_CAP,
            "pairs": pairs,
            "all_identical": all(row["identical"] for row in pairs),
            "pairs_sha256": digest(pairs),
        }
        (RAW / "recording_identity.json").write_text(json.dumps(identity, indent=2) + "\n")

    game, seed, cap = "ft09", SEEDS[0], 300
    main = _run_worker(game, seed, "main", cap)
    branch = _run_worker(game, seed, "off", cap)
    same = main["actions"] == branch["actions"] and main["trace_sha256"] == branch["trace_sha256"]
    differential = {
        "requirement": "REQ-ARC-WMTE-10016",
        "baseline_command": f"git show main:{MAIN_FILE}",
        "game": game,
        "seed": seed,
        "main": main,
        "branch": branch,
        "decisions_compared": min(main["actions"], branch["actions"]),
        "identical": same,
    }
    (RAW / "main_differential.json").write_text(json.dumps(differential, indent=2) + "\n")
    print(f"main differential: {same}, decisions={differential['decisions_compared']}", flush=True)
    if differential["decisions_compared"] < 300 or not same:
        raise AssertionError("main scored-policy differential failed")

    summary = {
        "requirement": "REQ-ARC-WMTE-10016",
        "inference_substrate": "offline_arcade_live_agent_runtime_self_discovery_no_llm",
        "cpu_only": True,
        "induction_disabled": True,
        "anonymous_policy_game_id": "unseen_runtime_game",
        "adapter_disabled": True,
        "banked_trajectories_disabled": True,
        "stored_engines_disabled": True,
        "scratch_cwd": "/tmp",
        "environment_files": str(resolve_environment_files(ROOT)),
        "identity_pairs": len(pairs),
        "identity_passed": identity["all_identical"],
        "main_decisions_compared": differential["decisions_compared"],
        "main_differential_passed": same,
        "proof_sha256": digest({"identity": identity, "differential": differential}),
    }
    (RAW / "proof_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--differential-only", action="store_true")
    parser.add_argument("--game")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--arm", choices=("on", "off", "main"))
    parser.add_argument("--cap", type=int)
    args = parser.parse_args()
    if args.worker:
        if args.game is None or args.seed is None or args.arm is None or args.cap is None:
            parser.error("worker requires game, seed, arm and cap")
        print(canonical(episode(args.game, args.seed, args.arm, args.cap)))
    else:
        print(canonical(run_proof(differential_only=args.differential_only)))


if __name__ == "__main__":
    main()

"""REQ-REPORT-8384: observe the production wrapper without any model or game recipe.

The observer records operands at the existing supervisor seam. It returns the
original decision, so recording cannot choose an action or supply a solution.
"""

from __future__ import annotations

import builtins
import importlib
from contextlib import contextmanager, ExitStack
from dataclasses import asdict
import json
import os
from pathlib import Path
import random
import socket
import subprocess
import time
from typing import Any, Iterator
from unittest.mock import patch

import numpy as np

from carnot.agentic import arc_competition_agent as agent_module
from carnot.agentic import arc_executable_world_model as world
from carnot.agentic import arc_frame_change_predictor as numeric
from carnot.agentic.arc_generalization_runtime import withheld_policy_inputs
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash

Json = dict[str, Any]
MODEL_SPECS: list[Json] = []
FORBIDDEN = (
    "llama_cpp",
    "transformers",
    "vllm",
    "mlx",
    "huggingface_hub",
    "sentence_transformers",
    "carnot.inference",
    "carnot.pipeline.gemma_loader",
    "carnot.agentic.arc_game_adapters",
)


@contextmanager
def no_models() -> Iterator[list[str]]:
    """Block imports, constructors, subprocesses and network before they can load a model."""
    attempts: list[str] = []
    original = builtins.__import__
    original_module = importlib.import_module

    def deny(*args: Any, **kwargs: Any) -> Any:
        attempts.append(str(args[0]) if args else "model_seam")
        raise RuntimeError("model_tripwire_before_load")

    def importing(name: str, *args: Any, **kwargs: Any) -> Any:
        fromlist = kwargs.get("fromlist", args[2] if len(args) > 2 else ()) or ()
        names = [name, *(name + "." + item for item in fromlist)]
        if any(
            candidate == prefix or candidate.startswith(prefix + ".")
            for candidate in names
            for prefix in FORBIDDEN
        ):
            return deny(name)
        return original(name, *args, **kwargs)

    def importing_module(name: str, package: str | None = None) -> Any:
        if any(name == prefix or name.startswith(prefix + ".") for prefix in FORBIDDEN):
            return deny(name)
        return original_module(name, package)

    with ExitStack() as stack:
        for target, name in [
            (builtins, "__import__"),
            (importlib, "import_module"),
            (world, "LocalGGUFProposer"),
            (agent_module.E3AgentPolicy, "_proposer"),
            (subprocess, "Popen"),
            (socket.socket, "connect"),
        ]:
            stack.enter_context(
                patch.object(
                    target,
                    name,
                    importing
                    if target is builtins
                    else importing_module
                    if target is importlib
                    else deny,
                )
            )
        yield attempts


def episode(unit: Json, arcade: Any, raw: Path) -> Json:
    """Run actual choose_action/is_done; constructed transports are used only by unit tests."""
    raw.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    frames: list[Any] = []
    latest = None
    error = None
    censoring: str | None = "action_limit"
    snapshots: list[Json] = []
    attempts: list[str] = []
    receipt: Json = {}
    actual_wrapper: Json = {}
    phase = "policy"
    count = 0

    def numeric_head_only() -> Any:
        # The normal loader also reads cached per-game transitions. Its existing
        # switch removes that memory while retaining the common numeric scorer.
        scorer = numeric.load_live_action_effect_scorer(root=agent_module.REPO, use_memory=False)
        agent_module._component_load_diagnostics.frame_change_scorer = dict(
            loaded=scorer is not None, cached_memory_withheld=True, small_numeric_head=True
        )
        return numeric.GroundTruthValidatedFrameChangeScorer(scorer) if scorer else None

    class Base:
        def __init__(self, game_id: str) -> None:
            self.game_id = game_id

    flags = {
        "CARNOT_ARC_TRAJECTORY_SUPERVISOR": str(int(unit["arm"] == "on")),
        "CARNOT_ARC_TRAJECTORY_SUPERVISOR_WINDOW": "120",
        "CARNOT_ARC_SUPERVISOR_TOOL_ARM": "0",
    }
    random.seed(unit["seed"])
    np.random.seed(unit["seed"])
    print(f"[exp8384] episode_before {unit['episode_id']}", flush=True)
    with (raw / "steps.jsonl").open("w") as stream:
        try:
            with (
                withheld_policy_inputs(),
                patch.dict(os.environ, flags),
                no_models() as attempts,
                patch.object(
                    agent_module, "_load_submitted_frame_change_scorer", numeric_head_only
                ),
                patch.object(agent_module, "load_solutions", lambda: {}),
            ):
                if os.environ.get("CARNOT_ARC_DISABLE_INDUCTION") != "1":
                    raise RuntimeError("induction_not_disabled")
                print("[exp8384] numeric_head_load_before no_LLM_load", flush=True)
                agent = agent_module.make_carnot_agent(Base, cascade=True, proposer=None)(
                    game_id=unit["game"]
                )
                print("[exp8384] numeric_head_load_after no_LLM_load", flush=True)
                policy = agent._policy
                actual_wrapper = dict(
                    factory="make_carnot_agent",
                    cascade=True,
                    policy=type(policy).__name__,
                    choose_action=0,
                    is_done=0,
                )
                supervisor = policy._trajectory_supervisor
                observing = supervisor.observe

                def observe(snapshot: Any) -> Any:
                    redirect = observing(snapshot)
                    snapshots.append(
                        dict(
                            snapshot=asdict(snapshot),
                            redirect=asdict(redirect) if redirect else None,
                        )
                    )
                    return redirect

                supervisor.observe = observe
                phase = "sdk"
                env = arcade.make(
                    unit["game"], seed=unit["seed"], scorecard_id=arcade.open_scorecard()
                )
                for index in range(unit["max_actions"]):
                    if time.monotonic() - started >= unit["max_seconds"]:
                        censoring = "time_limit"
                        break
                    phase = "policy"
                    actual_wrapper["is_done"] += 1
                    if agent.is_done(frames, latest):
                        censoring = None
                        break
                    before = time.monotonic()
                    action = agent.choose_action(frames, latest)
                    choice_s = time.monotonic() - before
                    actual_wrapper["choose_action"] += 1
                    if attempts:
                        raise RuntimeError("model_tripwire_observed")
                    if any(
                        a.get("skipped") != "disabled_by_env" for a in policy.induction_attempts
                    ):
                        raise RuntimeError("induction_executed")
                    payload = action.action_data.model_dump()
                    data = {k: v for k, v in payload.items() if k != "game_id"}
                    action_value = dict(name=action.name, data=data)
                    phase = "sdk"
                    before = time.monotonic()
                    latest = env.reset() if action.name == "RESET" else env.step(action, data=data)
                    sdk_s = time.monotonic() - before
                    if latest is None:
                        raise RuntimeError("sdk_null_observation")
                    visible = dict(
                        pixels=np.asarray(latest.frame).tolist(),
                        level=int(latest.levels_completed or 0),
                        available_actions=list(latest.available_actions),
                    )
                    row = dict(
                        index=index + 1,
                        action=action_value,
                        action_sha256=canonical_hash(action_value),
                        observation=visible,
                        observation_sha256=canonical_hash(visible),
                        progress=visible["level"] > (frames[-1].levels_completed if frames else 0),
                        choose_action_s=choice_s,
                        sdk_transition_s=sdk_s,
                        supervisor_events=list(snapshots),
                    )
                    snapshots.clear()
                    stream.write(json.dumps(row, sort_keys=True) + "\n")
                    stream.flush()
                    frames.append(latest)
                    count += 1
                    print(
                        f"[exp8384] {unit['episode_id']} completed={count} "
                        f"pending={unit['max_actions'] - count}",
                        flush=True,
                    )
                receipt = policy.trajectory_supervisor_diagnostics()
        except Exception as exc:
            error = f"{phase}:{type(exc).__name__}:{exc}"
            censoring = "exception"
            if actual_wrapper:
                receipt = policy.trajectory_supervisor_diagnostics()
    result = dict(
        unit=unit,
        error=error,
        censoring=censoring,
        duration_s=time.monotonic() - started,
        model_tripwire_passed=not attempts,
        model_tripwire_failed=bool(attempts),
        model_tripwire_attempts=attempts,
        supervisor_receipt=receipt,
        actual_wrapper_path=actual_wrapper,
        llm_invocation_count=0,
        pending_supervisor_events=snapshots,
        adapter_withheld=dict(
            stored_solutions=True,
            per_game_routes=True,
            per_game_learned_adapters=True,
            cached_action_effect_memory=True,
        ),
    )
    atomic_json(raw / "episode.json", result)
    print(
        f"[exp8384] episode_after {unit['episode_id']} completed={count} error={error}", flush=True
    )
    return result

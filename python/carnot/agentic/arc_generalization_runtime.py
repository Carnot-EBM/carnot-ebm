"""Adapter-withheld public ARC episode mechanics for REQ-ARC-WMTE-7708.

Only observations returned by the SDK cross into the scored agent. The
registry is read to disclose exposure; it never supplies actions or goals.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot.agentic import arc_competition_agent as competition_agent
from carnot.agentic import arc_executable_world_model as world


SALT = "v671-arc-20260926"
WITHHELD = (
    "per_game_adapter",
    "stored_route",
    "banked_solution",
    "game_source",
    "expert_goal",
    "offline_bfs_truth",
    "evaluator_label",
)


def game_digest(game: str) -> str:
    """Hash the frozen salt and public ID before seeing any outcome."""
    return hashlib.sha256(f"{SALT}{game}".encode()).hexdigest()


def freeze_schedule(roster: Iterable[str], registry: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Keep cleared games in the public roster while denying new solve credit."""
    games = sorted(set(roster), key=lambda game: (game_digest(game), game))
    if len(games) < 2:
        raise ValueError("two_sdk_games_required")
    known = {str(row.get("game")): row for row in registry}
    exposure = {
        game: {
            "levels_reproduced": int(known.get(game, {}).get("levels_reproduced") or 0),
            "full_game_clear": known.get(game, {}).get("full_game_clear") is True,
            "historically_public": True,
            "adapter_recorded": bool(known.get(game, {}).get("adapter")),
        }
        for game in games
    }
    return {
        "selection_rule": "smallest_sha256_of_salt_plus_game_id",
        "selection_salt": SALT,
        "roster": games,
        "registry_precheck": exposure,
        "rows": [
            {
                "episode_id": f"{game}:cpu_fixture",
                "game": game,
                "arm": "adapter_withheld",
                "max_actions": 3,
                "max_seconds": 30,
                "seed": 7708,
                "adapter_withheld": True,
                "withheld_inputs": list(WITHHELD),
                "prior_levels_reproduced": exposure[game]["levels_reproduced"],
            }
            for game in games[:2]
        ],
        "new_solve_credit": False,
        "claim_scope": "adapter-withheld first contact on historically exposed public games",
    }


@contextmanager
def withheld_policy_inputs() -> Any:
    """Deny stored policy routes and give the engine store a fresh empty home."""
    names = (
        "_recommend_live_approach",
        "load_cross_game_value_head",
        "_load_submitted_candidate_router",
        "_load_submitted_goal_energy_bias",
    )
    originals = {name: getattr(competition_agent, name) for name in names}
    original_route = competition_agent.arc_strategy_router.route_for_game

    def observed_route(game: str, *, mechanic: str | None = None, reg: Any = None) -> dict:
        # Registry game labels are withheld; a frame-derived mechanic may route.
        return original_route(game, mechanic=mechanic, reg={"games": []})

    old_dir = world.E3_DIR
    old_flags = {
        key: os.environ.get(key)
        for key in (
            "CARNOT_ARC_DISABLE_INDUCTION",
            "CARNOT_ARC_GOAL_PROBE_LOOP",
            "CARNOT_ARC_PROBE_PROTOCOL",
            "CARNOT_ARC_PLAYBOOK_RETRIEVAL",
            "CARNOT_ARC_PLAYBOOK_EXEMPLARS_ENABLED",
        )
    }
    with tempfile.TemporaryDirectory(prefix="exp7708-engine-") as empty:
        competition_agent.arc_strategy_router.route_for_game = observed_route
        competition_agent._recommend_live_approach = lambda game_id, mechanic=None: {
            "strategy": observed_route(game_id, mechanic=mechanic),
            "source": "observed_frame_or_default",
        }
        competition_agent.load_cross_game_value_head = lambda: None
        competition_agent._load_submitted_candidate_router = lambda *args, **kwargs: None
        competition_agent._load_submitted_goal_energy_bias = lambda: None
        world.E3_DIR = Path(empty)
        os.environ.update({key: "0" for key in old_flags})
        os.environ["CARNOT_ARC_DISABLE_INDUCTION"] = "1"
        try:
            yield
        finally:
            for name, value in originals.items():
                setattr(competition_agent, name, value)
            competition_agent.arc_strategy_router.route_for_game = original_route
            world.E3_DIR = old_dir
            for key, value in old_flags.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def _observation(frame: Any) -> dict[str, Any]:
    """Reduce visible SDK data without reading hidden game or evaluator labels."""
    visible = getattr(frame, "frame", None)

    def encode_array(value: Any) -> Any:
        if hasattr(value, "tolist") and hasattr(value, "shape"):
            return {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "pixels": value.tolist(),
            }
        raise TypeError(f"unsupported visible frame type: {type(value).__name__}")

    encoded = json.dumps(visible, sort_keys=True, default=encode_array).encode()
    return {
        "level": int(getattr(frame, "levels_completed", 0) or 0),
        "frame_sha256": hashlib.sha256(encoded).hexdigest(),
        "available_actions": list(getattr(frame, "available_actions", ()) or ()),
    }


def run_episode(
    unit: Mapping[str, Any],
    arcade: Any,
    *,
    agent_factory: Callable[..., type] | None = None,
) -> dict[str, Any]:
    """Run the actual scored wrapper against an SDK-compatible transport.

    The optional factory is for a small error fixture. Normal execution uses
    the production factory, and only SDK observations enter `choose_action`.
    """
    if agent_factory is None:
        agent_factory = competition_agent.make_carnot_agent

    class LocalAgentBase:
        def __init__(self, game_id: str) -> None:
            self.game_id = game_id

    game = str(unit["game"])
    started = time.monotonic()
    actions: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    frames: list[Any] = []
    latest = None
    counts = {"choose_action": 0, "sdk_transitions": 0, "observations": 0}
    censoring = None
    error = None
    policy_class = None
    sdk_inflight = False
    try:
        with withheld_policy_inputs():
            agent_type = agent_factory(LocalAgentBase, cascade=True, proposer=None)
            agent = agent_type(game_id=game)
            policy_class = type(agent._policy).__name__
            env = arcade.make(game, scorecard_id=arcade.open_scorecard())
            for index in range(int(unit["max_actions"])):
                if time.monotonic() - started >= float(unit.get("max_seconds", 30)):
                    censoring = "time_limit"
                    break
                if agent.is_done(frames, latest):
                    break
                before = time.monotonic()
                action = agent.choose_action(frames, latest)
                counts["choose_action"] += 1
                name = str(getattr(action, "name", action))
                payload = getattr(action, "action_data", None)
                data = payload.model_dump() if hasattr(payload, "model_dump") else None
                if isinstance(data, dict):
                    data = {key: value for key, value in data.items() if key != "game_id"}
                choice_s = time.monotonic() - before
                before = time.monotonic()
                sdk_inflight = True
                latest = env.reset() if name == "RESET" else env.step(action, data=data)
                if latest is None:
                    raise RuntimeError("sdk_null_observation")
                sdk_inflight = False
                counts["sdk_transitions"] += 1
                sdk_s = time.monotonic() - before
                observed = _observation(latest)
                counts["observations"] += 1
                observations.append(observed)
                attempts = getattr(agent._policy, "induction_attempts", [])
                last_attempt = attempts[-1] if attempts else {}
                actions.append(
                    {
                        "action_index": index + 1,
                        "action": name,
                        "data": data,
                        "actual_observation": observed,
                        "induction_attempt_count": len(attempts),
                        "induced_goal": last_attempt.get("goal_candidate_names"),
                        "model_acceptance": last_attempt.get("planned"),
                        "goal_firing": last_attempt.get("goal_fired"),
                        "sdk_transition": {"level_after": observed["level"]},
                        "resource_costs": {"choose_action_s": choice_s, "sdk_transition_s": sdk_s},
                    }
                )
                frames.append(latest)
                print(
                    f"[exp7708] episode={unit['episode_id']} completed={index + 1} "
                    f"elapsed_s={time.monotonic() - started:.3f}",
                    flush=True,
                )
    except Exception as exc:
        censoring = "sdk_exception" if sdk_inflight else "policy_exception"
        error = f"{type(exc).__name__}: {exc}"[:400]
    peak = max((row["level"] for row in observations), default=None)
    return {
        "episode_id": str(unit["episode_id"]),
        "game": game,
        "arm": str(unit.get("arm", "adapter_withheld")),
        "raw_metrics": {"peak_level": peak, "elapsed_s": time.monotonic() - started},
        "counts": counts,
        "exclusions": [],
        "censoring": censoring,
        "error": error,
        "provenance": "scripted_sdk_cpu_fixture",
        "policy_entry": {"factory": "make_carnot_agent", "policy_class": policy_class},
        "telemetry": actions,
        "solve_provenance": "development_proxy",
        "new_solve_credit": False,
    }

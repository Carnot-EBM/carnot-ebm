"""REQ-VICTRIS-9272-V789: offline Gymnasium-compatible Victris Arena wrapper aligned with arena-sim 0.83.0.

Enforces strict untrusted opponent shout stripping before observation exposure,
strict DEV_SEEDS (0..999) versus HELDOUT_SEEDS (10000..10999) disjointness and
split-membership refusal, legal turn enumeration without shout or tick fields,
and a hard <1000.0ms per-tick reply budget guard.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from carnot.victris.energy_bot import (
    UNTRUSTED_SHOUT_KEYS,
    extract_candidate_features,
    generate_candidate_turns,
    sanitize_observation,
    synthetic_tactical_observations,
)
from carnot.victris.rsi_loop import (
    DEV_SEEDS,
    HELDOUT_SEEDS,
    TICK_BUDGET_MS,
    enforce_tick_budget,
    validate_seed_split,
)


def legal_actions(obs: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Sanitize observation and enumerate legal candidate turns without shout or tick fields."""
    return generate_candidate_turns(sanitize_observation(obs))


class VictrisGymAdapter:
    """Offline Gymnasium-compatible wrapper for Victris Arena (arena-sim 0.83.0 contract)."""

    def __init__(
        self,
        env_factory: Callable[[str], Any] | None = None,
        opponent: str = "ts-starter",
        allowed_split: str = "dev",
        dev_seeds: Sequence[int] | None = None,
        heldout_seeds: Sequence[int] | None = None,
        tick_budget_ms: float = TICK_BUDGET_MS,
    ) -> None:
        self.dev_seeds = list(range(40)) if dev_seeds is None else [int(s) for s in dev_seeds]
        self.heldout_seeds = (
            list(range(10_000, 10_100))
            if heldout_seeds is None
            else [int(s) for s in heldout_seeds]
        )
        self.split_receipt = validate_seed_split(self.dev_seeds, self.heldout_seeds)
        if allowed_split not in {"dev", "heldout"}:
            raise ValueError("invalid_split_mode")
        if not (0.0 < float(tick_budget_ms) <= TICK_BUDGET_MS):
            raise ValueError("invalid_tick_budget")
        self.env_factory = env_factory
        self.opponent = str(opponent)
        self.allowed_split = allowed_split
        self.tick_budget_ms = float(tick_budget_ms)
        self._used_dev_seeds: set[int] = set()
        self._used_heldout_seeds: set[int] = set()
        self._env: Any = None
        self._current_obs: dict[str, Any] | None = None

    def sanitize_observation(self, obs: Mapping[str, Any]) -> dict[str, Any]:
        """Strip all untrusted opponent shout fields before feature extraction or decision making."""
        return sanitize_observation(obs)

    def legal_actions(self, obs: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
        """Return legal candidate turns for the given or current observation."""
        target = self._current_obs if obs is None else obs
        if target is None:
            raise ValueError("env_not_reset")
        return generate_candidate_turns(sanitize_observation(target))

    def reset(
        self, *, seed: int | None = None, options: Mapping[str, Any] | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset episode for an authorized split seed and return (sanitized_obs, info)."""
        _ = options
        if seed is None:
            raise ValueError("seed_required")
        s = int(seed)
        if self.allowed_split == "dev":
            if s not in DEV_SEEDS or s in self._used_heldout_seeds:
                raise ValueError("seed_split_violation")
            self._used_dev_seeds.add(s)
        else:
            if s not in HELDOUT_SEEDS or s in self._used_dev_seeds:
                raise ValueError("seed_split_violation")
            self._used_heldout_seeds.add(s)

        if self.env_factory is not None:
            self._env = self.env_factory(self.opponent)
            raw_obs, raw_info = self._env.reset(seed=s)
            hello = getattr(self._env, "_hello", {})
        else:
            scenarios = synthetic_tactical_observations()
            raw_obs = dict(scenarios[s % len(scenarios)]["obs"], tick=0)
            raw_info = {}
            hello = {"arena": "synthetic_tactical_fixture", "opponent": self.opponent}

        clean_obs = sanitize_observation(raw_obs)
        self._current_obs = clean_obs
        info = {
            "hello": dict(hello) if isinstance(hello, Mapping) else {},
            "legal_actions": generate_candidate_turns(clean_obs),
            "seed": s,
            "split": self.allowed_split,
            **(dict(raw_info) if isinstance(raw_info, Mapping) else {}),
        }
        return clean_obs, info

    def step(
        self, action: Mapping[str, Any] | int, elapsed_ms: float | None = None
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Step environment with a legal turn dict or action index after enforcing <1000ms tick budget."""
        if self._current_obs is None:
            raise ValueError("env_not_reset")
        enforce_tick_budget(0.0 if elapsed_ms is None else float(elapsed_ms), self.tick_budget_ms)

        if isinstance(action, int) and not isinstance(action, bool):
            cands = generate_candidate_turns(self._current_obs)
            if action < 0 or action >= len(cands):
                raise ValueError("invalid_action")
            turn = dict(cands[action])
        elif isinstance(action, Mapping):
            if any(k in action for k in UNTRUSTED_SHOUT_KEYS):
                raise ValueError("untrusted_shout_in_action")
            turn = {k: v for k, v in action.items() if k in {"move", "dash", "attack"}}
        else:
            raise ValueError("invalid_action")

        if self._env is not None:
            raw_next, reward, terminated, truncated, raw_info = self._env.step(
                {**turn, "tick": self._current_obs["tick"]}
            )
        else:
            feats = extract_candidate_features(self._current_obs, turn)
            reward = float(feats[8] + feats[9] - 0.5 * feats[3])
            raw_next = dict(self._current_obs, tick=int(self._current_obs["tick"]) + 1)
            terminated = True
            truncated = False
            raw_info = {"end": {"result": "win" if reward >= 0.5 else "loss"}}

        clean_next = sanitize_observation(raw_next)
        self._current_obs = clean_next
        info = {
            "legal_actions": generate_candidate_turns(clean_next),
            "split": self.allowed_split,
            **(dict(raw_info) if isinstance(raw_info, Mapping) else {}),
        }
        return clean_next, float(reward), bool(terminated), bool(truncated), info

    def close(self) -> None:
        """Close underlying simulator instance if open."""
        if self._env is not None:
            self._env.close()
            self._env = None
        self._current_obs = None

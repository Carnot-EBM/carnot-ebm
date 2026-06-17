"""ARC-AGI-3 reusable offline solver kit — the durable scaffolding so that what
we learn solving one game is REUSED on the next, and every solve is captured as
OFFLINE-REPRODUCIBLE (not a live-recorded coordinate trajectory that silently
rots).

Why this module exists
----------------------
2026-06-16: making the deeper sc25/lp85 levels offline-reproducible revealed that
the prior solves were banked as live-recorded `solve_trace.actions` whose pixel
coordinates were coupled to the LIVE env layout, so they replay to 0 levels on
the offline `environment_files` env. The effort was effectively wasted because
the WINNING CONDITION (the search that derives a solution for the actual env) was
never captured — only one frozen trajectory was. This kit captures the general,
reusable primitives + the hard-won per-game gotchas so future games plug in their
action-model + win-check and inherit the rest, and so every solve passes a
reproduction gate before it counts.

Hard-won general gotchas (apply to ANY ARC-AGI-3 game; see ops/arc_solve_registry.yaml)
-----------------------------------------------------------------------------------
1. OFFLINE is a deterministic simulator: `Arcade(OperationMode.OFFLINE,
   environments_dir=environment_files)` loads all 25 games, zero network/quota.
2. The LEVEL lives on the FRAME (`frame.levels_completed`), NOT on `env._game`.
3. `env._game = copy.deepcopy(state)` injection works for SOME games (lp85) but is
   BROKEN for others (sc25) — references don't survive deepcopy. The robust,
   universal approach is REPLAY-FROM-RESET (operate on the real env).
4. The FIRST `env.step` after `env.reset()` is CONSUMED (no-op) in at least sc25.
   Always do a warm-up step after reset before applying a path.
5. Element COORDINATES must be DISCOVERED from the env, never hardcoded — the live
   solver's hardcoded coords (e.g. sc25 SC25_GRID_COORDS) miss the offline layout.
   Use env-adaptive discovery (cf. lp85 `discover_click_buttons`; sc25 camera is
   identity so cell (r,c) is at display (24+5c, 49+5r)).
6. Some games have STATE-DEPENDENT controls (sc25 tank-controls: press-new-
   direction turns, press-same moves) and MULTI-FRAME ANIMATIONS that must be let
   to resolve — the dedup state-key MUST include facing/phase, and you must step
   until animation phase flags clear before the next action / win check.
"""
from __future__ import annotations

import heapq
import itertools
from pathlib import Path
from typing import Any, Callable, Hashable, Optional, Sequence

REPO = Path(__file__).resolve().parents[3]
ENV_DIR = REPO / "environment_files"


def offline_arcade() -> Any:  # pragma: no cover - thin SDK boundary
    """A zero-quota, no-network OFFLINE Arcade over the local environment_files."""
    from arc_agi import Arcade
    from arc_agi.base import OperationMode

    return Arcade(arc_api_key="", operation_mode=OperationMode.OFFLINE,
                  environments_dir=str(ENV_DIR))


def frame_level(frame: Any) -> int:
    """The level is read from the FRAME, never from env._game (gotcha #2)."""
    if frame is None:
        return -1
    return int(getattr(frame, "levels_completed", 0) or 0)


class OfflineSolver:
    """Reusable from-scratch offline ARC solver (replay-from-reset BFS).

    Plug in a per-game model; inherit the universal harness (offline arcade,
    warm-up after reset, replay-from-reset, frame-based level, BFS + dedup,
    level chaining). Each game supplies:
      - action_labels(env) -> list[str]: the action vocabulary at the current
        state (env-DISCOVERED, not hardcoded; gotcha #5).
      - apply(env, label, frame) -> frame: execute one action, resolving any
        animation (gotcha #6), and return the new frame.
      - state_key(game) -> Hashable: the dedup key — MUST include every
        load-bearing piece of state (position, FACING, cast-state, sprites;
        gotcha #6), else turns/no-ops collapse and the search stalls.
    """

    def __init__(self, game_id: str, action_labels: Callable[[Any], Sequence[str]],
                 apply: Callable[[Any, str, Any], Any], state_key: Callable[[Any], Hashable],
                 *, warmup_label: Optional[str] = None, max_nodes: int = 30000,
                 verifier: Optional[Callable[[Any], float]] = None) -> None:
        self.game_id = game_id
        self.action_labels = action_labels
        self.apply = apply
        self.state_key = state_key
        self.warmup_label = warmup_label  # an action to consume the no-op first slot (gotcha #4)
        self.max_nodes = max_nodes
        # VERIFIER-ROUTED SEARCH (the north-star efficiency loop): a score on a
        # state (LOWER = closer to the win, an energy/goal-distance). When given,
        # the search is best-first ordered by it, so it expands promising branches
        # first and the state count SHRINKS. When None, it degrades to plain BFS
        # (verifier ≡ 0 → the heap orders by insertion = FIFO). Pass a learned or
        # computed verifier to turn the solver into a verifier-routed search.
        self.verifier = verifier or (lambda _g: 0.0)
        self.last_states_expanded = 0
        self.last_frame: Any = None

    def _call_state_key(self, env: Any) -> Hashable:
        try:
            return self.state_key(env._game, self.last_frame)  # type: ignore[misc]
        except TypeError:
            return self.state_key(env._game)

    def _call_verifier(self, env: Any) -> float:
        try:
            return float(self.verifier(env._game, self.last_frame))  # type: ignore[misc]
        except TypeError:
            return float(self.verifier(env._game))

    def _call_action_labels(self, env: Any, path: Sequence[str]) -> Sequence[str]:
        try:
            return self.action_labels(env, self.last_frame, tuple(path))  # type: ignore[misc]
        except TypeError:
            try:
                return self.action_labels(env, self.last_frame)  # type: ignore[misc]
            except TypeError:
                return self.action_labels(env)

    def _replay(self, env: Any, path: Sequence[str]) -> Any:
        f = env.reset()
        self.last_frame = f
        if self.warmup_label is not None:
            f = self.apply(env, self.warmup_label, f)  # gotcha #4
            self.last_frame = f
        for label in path:
            f = self.apply(env, label, f)
            self.last_frame = f
        return f

    def solve_level(self, env: Any, start_level: int, prefix: Sequence[str], depth_cap: int):
        """Search one level forward from `prefix` — verifier-routed BEST-FIRST (or
        plain BFS when no verifier). Returns (extension_path, states_expanded)."""
        self._replay(env, list(prefix))
        seen = {self._call_state_key(env)}
        counter = itertools.count()  # FIFO tiebreaker (so verifier≡0 ⇒ BFS)
        heap = [(self._call_verifier(env), next(counter), [])]
        nodes = 0
        while heap and nodes < self.max_nodes:
            _, _, path = heapq.heappop(heap)
            if len(path) >= depth_cap:
                continue
            self._replay(env, list(prefix) + path)
            for label in self._call_action_labels(env, path):
                f2 = self.apply(env, label, None)
                self.last_frame = f2
                nodes += 1
                if frame_level(f2) > start_level:
                    self.last_states_expanded = nodes
                    return path + [label], nodes
                k = self._call_state_key(env)
                if k not in seen:
                    seen.add(k)
                    heapq.heappush(heap, (self._call_verifier(env), next(counter), path + [label]))
                self._replay(env, list(prefix) + path)  # restore for next sibling
        self.last_states_expanded = nodes
        return None, nodes

    def solve(self, env: Any, target_level: int, depth_cap: int = 30):
        """Chain levels from reset to target_level; return the full action path + reached level."""
        f = self._replay(env, [])
        cur = frame_level(f)
        full: list[str] = []
        for lvl in range(cur + 1, target_level + 1):
            path, _ = self.solve_level(env, cur, full, depth_cap)
            if path is None:
                break
            f = self._replay(env, full + path)
            cur = frame_level(f)
            full += path
            if cur < lvl:
                break
        return full, cur


def reproduce(game_id: str, solution: Sequence[str], apply: Callable[[Any, str, Any], Any],
              *, warmup_label: Optional[str] = None, claimed_level: Optional[int] = None) -> dict:
    """THE REPRODUCTION GATE. Replay a banked `solution` against the OFFLINE env and
    report the level it actually reaches. A solve is only real if this reproduces
    the claimed level offline — never trust a live-recorded trajectory alone.

    Returns {reached_level, claimed_level, reproduced: bool}. Zero quota.
    """
    arc = offline_arcade()
    env = arc.make(game_id, scorecard_id=arc.open_scorecard())
    f = env.reset()
    if warmup_label is not None:
        f = apply(env, warmup_label, f)
    for label in solution:
        f = apply(env, label, f)
    reached = frame_level(f)
    return {
        "game": game_id,
        "reached_level": reached,
        "claimed_level": claimed_level,
        "reproduced": (claimed_level is None) or (reached >= int(claimed_level)),
        "mode": "offline_reproduction_gate_no_quota",
    }

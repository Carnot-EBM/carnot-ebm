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
7. Some config/toggle games call next_level() in the SAME action that creates the
   winning arrangement, so the returned frame is already the next level and the
   pre-win grid is not externally observable. Ground such win predicates on the
   execution state immediately before next_level, then count only the reproduce()
   level advance.
"""
from __future__ import annotations

import copy
import heapq
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Hashable, Optional, Sequence

REPO = Path(__file__).resolve().parents[3]
ENV_DIR = REPO / "environment_files"
ARC_STANDING_PATH_COST_WEIGHT = 1.0
ARC_BASELINE_PATH_COST_WEIGHT = 0.0


@dataclass(frozen=True)
class PrimitiveOperator:
    """A reusable ARC solve operator learned from one or more reproduced games."""

    operator: str
    derived_from_games: tuple[str, ...]
    purpose: str
    selector_tags: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "operator": self.operator,
            "derived_from_games": list(self.derived_from_games),
            "purpose": self.purpose,
            "selector_tags": list(self.selector_tags),
        }


def primitive_operator_registry() -> tuple[PrimitiveOperator, ...]:
    """REQ-REPORT-4436: consolidated generic operators available to the standing loop."""

    return (
        PrimitiveOperator(
            operator="glyph_rewrite_matcher",
            derived_from_games=("tr87",),
            purpose="Greedy multi-glyph LHS->RHS rewrite, including repeated passes.",
            selector_tags=("config_substitution", "glyph", "rewrite", "tr87"),
        ),
        PrimitiveOperator(
            operator="config_rule_grounding",
            derived_from_games=("s5i5", "ft09", "tr87"),
            purpose="Ground a proposed config rule against predicted object/register coverage.",
            selector_tags=("config_toggle", "marker_coverage", "local_constraint", "rule"),
        ),
        PrimitiveOperator(
            operator="graph_astar_action_cost",
            derived_from_games=("tu93", "lp85", "cd82", "sp80", "cn04", "m0r0", "sk48", "su15"),
            purpose="A* frontier priority: standing path cost plus verifier/action-cost heuristic.",
            selector_tags=("graph_explore", "astar", "action_cost", "keyboard", "click"),
        ),
        PrimitiveOperator(
            operator="object_centric_digest",
            derived_from_games=("g50t", "lp85", "tn36", "ka59"),
            purpose="Connected-component object summary for routing, grounding, and active data.",
            selector_tags=("object", "digest", "program_editor", "world_model"),
        ),
        PrimitiveOperator(
            operator="active_data_collection",
            derived_from_games=("ar25", "ka59", "ft09", "sc25"),
            purpose="Balanced action/object coverage plan for offline transition collection.",
            selector_tags=("active_data", "world_model", "e3", "transition"),
        ),
    )


def select_primitive_operators(
    *, mechanic_class: Optional[str] = None, action_model: str = "", game: str = ""
) -> tuple[PrimitiveOperator, ...]:
    """Select generic operators before per-game reverse engineering.

    This is intentionally conservative: it exposes reusable operators for the standing
    loop without removing any per-game adapter path.
    """

    registry = {op.operator: op for op in primitive_operator_registry()}
    mechanic = (mechanic_class or "").lower()
    action = (action_model or "").lower()
    gid = (game or "").lower()

    if "config_substitution" in mechanic or "glyph" in mechanic or gid == "tr87":
        names = ("glyph_rewrite_matcher", "graph_astar_action_cost", "object_centric_digest")
    elif "config" in mechanic or "toggle" in mechanic or "constraint" in mechanic:
        names = ("config_rule_grounding", "object_centric_digest", "graph_astar_action_cost")
    elif "program_editor" in mechanic:
        names = ("object_centric_digest", "active_data_collection", "graph_astar_action_cost")
    elif "world_model" in mechanic or "e3" in mechanic:
        names = ("active_data_collection", "object_centric_digest", "graph_astar_action_cost")
    elif "keyboard" in action or "click" in action or "graph" in mechanic:
        names = ("graph_astar_action_cost", "object_centric_digest")
    else:
        names = ("object_centric_digest", "active_data_collection", "graph_astar_action_cost")
    return tuple(registry[name] for name in names)


def cyclic_distance(current: int, target: int, *, modulus: int = 7) -> int:
    """Shortest cyclic distance on an integer wheel, used by config/glyph solvers."""

    if modulus <= 0:
        raise ValueError("modulus must be positive")
    return min((int(target) - int(current)) % modulus, (int(current) - int(target)) % modulus)


def sequence_cyclic_distance(
    current: Sequence[int],
    required: Sequence[int],
    *,
    modulus: int = 7,
    gap_cost: Optional[float] = None,
) -> float:
    """Sum cyclic distance over aligned values, with a bounded length-gap penalty."""

    n = min(len(current), len(required))
    gap = float(modulus if gap_cost is None else gap_cost)
    return float(
        sum(cyclic_distance(current[i], required[i], modulus=modulus) for i in range(n))
        + gap * abs(len(current) - len(required))
    )


def greedy_rewrite(
    sequence: Sequence[Hashable],
    rules: Sequence[tuple[Sequence[Hashable], Sequence[Hashable]]],
    *,
    passes: int = 1,
) -> tuple[Hashable, ...] | None:
    """Greedy first-prefix LHS->RHS rewrite, repeated for tr87-style chains."""

    normalized = [(tuple(lhs), tuple(rhs)) for lhs, rhs in rules]
    out: tuple[Hashable, ...] = tuple(sequence)
    for _ in range(max(0, int(passes))):
        pos = 0
        rewritten: list[Hashable] = []
        while pos < len(out):
            for lhs, rhs in normalized:
                if out[pos:pos + len(lhs)] == lhs:
                    rewritten.extend(rhs)
                    pos += len(lhs)
                    break
            else:
                return None
        out = tuple(rewritten)
    return out


def ground_marker_coverage_rule(
    *,
    controlled_markers: Sequence[tuple[int, int]],
    target_markers: Sequence[tuple[int, int]],
    step: int,
    horizontal_label: str,
    vertical_label: str,
) -> dict[str, Any]:
    """Derive a config-toggle path from a grounded marker-coverage predicate."""

    predicted = [tuple(marker) for marker in controlled_markers]
    path: list[str] = []
    for target_x, target_y in target_markers:
        candidates = [(i, x, y) for i, (x, y) in enumerate(predicted) if y == target_y and x < target_x]
        if candidates:
            index, x, _ = candidates[0]
            moves = (target_x - x) // int(step)
            path.extend([horizontal_label] * moves)
            predicted[index] = (target_x, target_y)
    for target_x, target_y in target_markers:
        candidates = [(i, x, y) for i, (x, y) in enumerate(predicted) if x == target_x and y < target_y]
        if candidates:
            index, _, y = candidates[0]
            moves = (target_y - y) // int(step)
            path.extend([vertical_label] * moves)
            predicted[index] = (target_x, target_y)
    satisfied = all(tuple(target) in predicted for target in target_markers)
    return {
        "operator": "config_rule_grounding",
        "solution": path,
        "predicted_markers": [tuple(marker) for marker in predicted],
        "target_markers": [tuple(marker) for marker in target_markers],
        "predicate_satisfied": satisfied,
    }


def object_centric_digest(grid: Any) -> dict[str, Any]:
    """Connected-component digest for ARC frames or grids."""

    import numpy as np

    arr = np.asarray(grid)
    if arr.ndim != 2:
        raise ValueError("object_centric_digest expects a 2-D grid")
    vals, counts = np.unique(arr, return_counts=True)
    background = int(vals[counts.argmax()]) if len(vals) else 0
    mask = arr != background
    seen = np.zeros_like(mask, dtype=bool)
    components: list[dict[str, Any]] = []
    h, w = arr.shape
    for y0 in range(h):
        for x0 in range(w):
            if not mask[y0, x0] or seen[y0, x0]:
                continue
            color = int(arr[y0, x0])
            stack = [(y0, x0)]
            seen[y0, x0] = True
            cells: list[tuple[int, int]] = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if (
                        0 <= ny < h
                        and 0 <= nx < w
                        and mask[ny, nx]
                        and not seen[ny, nx]
                        and int(arr[ny, nx]) == color
                    ):
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            ys = [y for y, _ in cells]
            xs = [x for _, x in cells]
            bbox = [min(ys), min(xs), max(ys), max(xs)]
            area = len(cells)
            signature = f"c{color}:a{area}:bbox{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}"
            components.append(
                {
                    "color": color,
                    "area": area,
                    "bbox": bbox,
                    "centroid": [sum(xs) / area, sum(ys) / area],
                    "signature": signature,
                }
            )
    components.sort(key=lambda row: (-int(row["area"]), int(row["color"]), row["bbox"]))
    return {
        "operator": "object_centric_digest",
        "shape": [int(arr.shape[0]), int(arr.shape[1])],
        "background_color": background,
        "component_count": len(components),
        "components": components,
    }


def active_data_collection_plan(
    *,
    action_labels: Sequence[str],
    object_signatures: Sequence[str],
    max_cases_per_action: int = 3,
) -> list[dict[str, Any]]:
    """Balanced action/object coverage rows for offline transition collection."""

    signatures = list(object_signatures) or ["none"]
    rows: list[dict[str, Any]] = []
    for action in action_labels:
        for case_index in range(max(0, int(max_cases_per_action))):
            rows.append(
                {
                    "operator": "active_data_collection",
                    "action": str(action),
                    "object_signature": signatures[case_index % len(signatures)],
                    "case_index": case_index,
                    "selection_policy": "balanced_action_object_coverage",
                }
            )
    return rows


def astar_frontier_priority(
    *, depth: int, heuristic: float, path_cost_weight: Optional[float] = None
) -> float:
    """Standing graph/A* priority shared by OfflineSolver and graph-explore users."""

    return float(standing_path_cost_weight(path_cost_weight) * int(depth) + float(heuristic))


def standing_path_cost_weight(path_cost_weight: Optional[float]) -> float:
    """REQ-LEARN-4364: default ARC planning to additive A* cost; keep 0.0 as baseline."""
    if path_cost_weight is None:
        return ARC_STANDING_PATH_COST_WEIGHT
    return float(path_cost_weight)


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
                 verifier: Optional[Callable[[Any], float]] = None,
                 path_cost_weight: Optional[float] = None, branch_mode: str = "replay",
                 env_factory: Optional[Callable[[], Any]] = None) -> None:
        self.game_id = game_id
        self.action_labels = action_labels
        self.apply = apply
        self.state_key = state_key
        self.warmup_label = warmup_label  # an action to consume the no-op first slot (gotcha #4)
        self.max_nodes = max_nodes
        # BRANCH MODE — how the search navigates between nodes:
        #   "replay"  (default, UNCHANGED): replay-from-reset (env.reset() + re-apply the path) per
        #             node. Correct + memory-light for games whose state is fully a function of the
        #             action prefix from reset (lp85, sc25). The proven default; do not change it.
        #   "deepcopy": snapshot copy.deepcopy(env._game) per node and restore by deepcopy, branching
        #             the EXACT env state rather than reconstructing it. Use for games where
        #             replay-from-reset does not faithfully reproduce the searched state (the verifier
        #             then finds a path that fails the reproduction gate). Costs a deepcopy per node;
        #             requires env._game to be deepcopy-able + injectable — the deepcopy-injection
        #             gotcha #3 means it is NOT universal (works for lp85, BROKEN for sc25/tu93).
        #   "fresh_env": make a BRAND-NEW env (env_factory, default a fresh arc.make of game_id) for
        #             EVERY candidate evaluation and replay prefix+path from reset on it. The fix for a
        #             game whose env.reset() is NON-IDEMPOTENT (gotcha #7: tu93's reset leaves a
        #             parity-toggling hidden state, so the reuse-one-env search detects parity-contingent
        #             "wins" that fail the fresh-env reproduction gate). A fresh env always starts at the
        #             SAME pristine parity the gate uses, so found paths reproduce. Costs a fresh env +
        #             full replay per evaluation — slower, so reserve it for non-idempotent-reset games.
        self.branch_mode = branch_mode
        self.env_factory = env_factory       # () -> fresh env, for branch_mode="fresh_env"
        self._fresh_arcade: Any = None       # lazily-cached arcade + scorecard for the default factory
        self._fresh_scorecard: Any = None
        # VERIFIER-ROUTED SEARCH (the north-star efficiency loop): a score on a
        # state (LOWER = closer to the win, an energy/goal-distance). When given,
        # the search is best-first ordered by it, so it expands promising branches
        # first and the state count SHRINKS. When None, it degrades to plain BFS
        # (verifier ≡ 0 → the heap orders by insertion = FIFO). Pass a learned or
        # computed verifier to turn the solver into a verifier-routed search.
        self.verifier = verifier or (lambda _g: 0.0)
        self.path_cost_weight = standing_path_cost_weight(path_cost_weight)
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

    def _priority(self, env: Any, path: Sequence[str]) -> float:
        """Verifier score plus standing path cost. Pass 0.0 for legacy greedy routing."""
        return astar_frontier_priority(
            depth=len(path),
            heuristic=self._call_verifier(env),
            path_cost_weight=self.path_cost_weight,
        )

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
        if self.branch_mode == "deepcopy":
            return self._solve_level_deepcopy(env, start_level, prefix, depth_cap)
        if self.branch_mode == "fresh_env":
            return self._solve_level_fresh(env, start_level, prefix, depth_cap)
        self._replay(env, list(prefix))
        seen = {self._call_state_key(env)}
        counter = itertools.count()  # FIFO tiebreaker (so verifier≡0 ⇒ BFS)
        heap = [(self._priority(env, []), next(counter), [])]
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
                    child_path = path + [label]
                    heapq.heappush(heap, (self._priority(env, child_path), next(counter), child_path))
                self._replay(env, list(prefix) + path)  # restore for next sibling
        self.last_states_expanded = nodes
        return None, nodes

    def _solve_level_deepcopy(self, env: Any, start_level: int, prefix: Sequence[str], depth_cap: int):
        """DEEPCOPY-PER-NODE variant of solve_level. Instead of replaying-from-reset to navigate, it
        SNAPSHOTS copy.deepcopy(env._game) per node and restores by deepcopy — branching the EXACT env
        state (incl. anything replay-from-reset doesn't faithfully reconstruct). Each heap node carries
        its (snapshot, frame) so state_key/verifier see the right frame; the found path is identical in
        shape to the replay variant (a sequence of labels) so the reproduction gate is unchanged."""
        self._replay(env, list(prefix))
        seen = {self._call_state_key(env)}
        counter = itertools.count()
        root = (self._priority(env, []), next(counter), [], copy.deepcopy(env._game), self.last_frame)
        heap = [root]
        nodes = 0
        while heap and nodes < self.max_nodes:
            _, _, path, snap, frame = heapq.heappop(heap)
            if len(path) >= depth_cap:
                continue
            env._game = copy.deepcopy(snap)            # restore this node's exact state
            self.last_frame = frame
            for label in self._call_action_labels(env, path):
                env._game = copy.deepcopy(snap)        # branch from the node for each child
                self.last_frame = frame
                f2 = self.apply(env, label, None)
                self.last_frame = f2
                nodes += 1
                if frame_level(f2) > start_level:
                    self.last_states_expanded = nodes
                    return path + [label], nodes
                k = self._call_state_key(env)
                if k not in seen:
                    seen.add(k)
                    child_path = path + [label]
                    heapq.heappush(heap, (self._priority(env, child_path), next(counter),
                                          child_path, copy.deepcopy(env._game), f2))
        self.last_states_expanded = nodes
        return None, nodes

    def _fresh_env(self) -> Any:
        """A BRAND-NEW env for branch_mode='fresh_env' — pristine reset parity. Default: a fresh
        arc.make of self.game_id over a lazily-cached offline arcade + scorecard."""
        if self.env_factory is not None:
            return self.env_factory()
        if self._fresh_arcade is None:
            self._fresh_arcade = offline_arcade()
            self._fresh_scorecard = self._fresh_arcade.open_scorecard()
        return self._fresh_arcade.make(self.game_id, scorecard_id=self._fresh_scorecard)

    def _solve_level_fresh(self, env: Any, start_level: int, prefix: Sequence[str], depth_cap: int):
        """FRESH-ENV-PER-NODE variant of solve_level. EVERY candidate is evaluated on a BRAND-NEW env
        (replay prefix+path from reset), so each evaluation sees the same pristine reset parity the
        reproduction gate uses — the fix for non-idempotent-reset games (gotcha #7: a game whose
        env.reset() leaves parity-toggling hidden state, where the reuse-one-env search detects
        parity-contingent wins that fail the fresh-env gate). The `env` arg is unused — the factory
        mints fresh envs. Slower (a fresh env + full replay per evaluation); reserve for such games."""

        def at(path: Sequence[str]):
            e = self._fresh_env()
            self._replay(e, list(prefix) + list(path))   # reset+replay on the fresh env; sets last_frame
            return e

        e0 = at([])
        seen = {self._call_state_key(e0)}
        counter = itertools.count()
        heap = [(self._priority(e0, []), next(counter), [])]
        nodes = 0
        while heap and nodes < self.max_nodes:
            _, _, path = heapq.heappop(heap)
            if len(path) >= depth_cap:
                continue
            e_node = at(path)                             # fresh env at the node (for action_labels)
            for label in self._call_action_labels(e_node, path):
                e_child = at(path + [label])
                f2 = self.last_frame
                nodes += 1
                if frame_level(f2) > start_level:
                    self.last_states_expanded = nodes
                    return path + [label], nodes
                k = self._call_state_key(e_child)
                if k not in seen:
                    seen.add(k)
                    heapq.heappush(heap, (self._priority(e_child, path + [label]),
                                          next(counter), path + [label]))
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
